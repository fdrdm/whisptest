import os
import shutil
import hashlib
import tempfile
import time
from pathlib import Path
from fastapi import FastAPI, Form, UploadFile, File as FastApiFile, Request
from fastapi.responses import JSONResponse
import gradio as gr
from gradio_client import utils as client_utils
import uvicorn

CHUNK_SIZE = 20 * 1024 * 1024  # 20 MB por parte (seguro para el límite de 90-100MB de Cloudflare)
CHUNK_TEMP_DIR = Path(tempfile.gettempdir()) / "whisptest_upload_chunks"
CHUNK_TEMP_DIR.mkdir(parents=True, exist_ok=True)

def cleanup_stale_chunks():
    """Elimina fragmentos huérfanos de subidas canceladas con más de 2 horas."""
    now = time.time()
    try:
        for item in CHUNK_TEMP_DIR.iterdir():
            if item.is_dir() and (now - item.stat().st_mtime) > 7200:
                shutil.rmtree(str(item), ignore_errors=True)
    except Exception:
        pass

HEAD_HTML = """
<style>
/* 1. Ajuste automático de texto y celdas en las tablas de Gradio */
.gradio-dataframe table,
.gradio-dataframe table td,
.gradio-dataframe table th {
    white-space: normal !important;
    word-break: break-word !important;
    overflow-wrap: anywhere !important;
    vertical-align: middle !important;
    line-height: 1.35 !important;
}

.gradio-dataframe .table-wrap {
    overflow-x: auto !important;
}

/* 2. Toast de progreso de subida por partes para Cloudflare Tunnel */
#cf-chunk-upload-toast {
    position: fixed;
    bottom: 24px;
    right: 24px;
    z-index: 999999;
    background: #141414;
    color: #f4faf8;
    border-left: 3px solid #038169;
    padding: 16px 20px;
    border-radius: 12px;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5), 0 0 0 1px rgba(255, 255, 255, 0.1);
    font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    font-size: 13px;
    min-width: 320px;
    max-width: 440px;
    display: none;
    transition: opacity 0.3s ease;
}
</style>

<script>
(function() {
    const CHUNK_SIZE = 20 * 1024 * 1024; // 20 MB por parte
    const originalFetch = window.fetch;

    function getOrCreateToast() {
        let toast = document.getElementById('cf-chunk-upload-toast');
        if (!toast) {
            toast = document.createElement('div');
            toast.id = 'cf-chunk-upload-toast';
            document.body.appendChild(toast);
        }
        return toast;
    }

    function updateToastProgress(current, total, filename) {
        const toast = getOrCreateToast();
        const pct = Math.round((current / total) * 100);
        toast.style.display = 'block';
        toast.style.opacity = '1';
        toast.innerHTML = `
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                <span style="font-weight: 600; font-size: 13px; color: #38bdf8;">⚡ Subiendo por partes (Cloudflare)</span>
                <span style="font-weight: 700; color: #38bdf8;">${pct}%</span>
            </div>
            <div style="color: #cbd5e1; font-size: 12px; margin-bottom: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;" title="${filename}">
                📁 ${filename}
            </div>
            <div style="font-size: 11px; color: #94a3b8; margin-bottom: 8px;">
                Parte <strong>${current}</strong> de <strong>${total}</strong> (límite 90MB respetado)
            </div>
            <div style="background: #334155; border-radius: 6px; height: 6px; overflow: hidden;">
                <div style="background: linear-gradient(90deg, #38bdf8, #818cf8); width: ${pct}%; height: 100%; transition: width 0.25s ease;"></div>
            </div>
        `;
    }

    function hideToastStatus(isError, message) {
        const toast = getOrCreateToast();
        if (isError) {
            toast.innerHTML = `
                <div style="color: #f87171; font-weight: 600;">
                    ❌ Error al subir: ${message}
                </div>
            `;
            setTimeout(() => {
                toast.style.opacity = '0';
                setTimeout(() => { toast.style.display = 'none'; }, 300);
            }, 6000);
        } else {
            toast.innerHTML = `
                <div style="color: #34d399; font-weight: 600; display: flex; align-items: center; gap: 6px;">
                    <span>✅ Subida completa (${message})</span>
                </div>
            `;
            setTimeout(() => {
                toast.style.opacity = '0';
                setTimeout(() => { toast.style.display = 'none'; }, 300);
            }, 3000);
        }
    }

    async function uploadSingleFileInChunks(file) {
        const totalChunks = Math.ceil(file.size / CHUNK_SIZE);
        const uploadId = 'up_' + Date.now() + '_' + Math.random().toString(36).substring(2, 9);
        const fileName = file.name || 'archivo';
        let finalDest = null;

        for (let i = 0; i < totalChunks; i++) {
            const start = i * CHUNK_SIZE;
            const end = Math.min(file.size, start + CHUNK_SIZE);
            const chunkBlob = file.slice(start, end);

            updateToastProgress(i + 1, totalChunks, fileName);

            const chunkFormData = new FormData();
            chunkFormData.append('upload_id', uploadId);
            chunkFormData.append('chunk_index', i.toString());
            chunkFormData.append('total_chunks', totalChunks.toString());
            chunkFormData.append('filename', fileName);
            chunkFormData.append('chunk', chunkBlob, fileName);

            const res = await originalFetch('/upload_chunk', {
                method: 'POST',
                body: chunkFormData,
                credentials: 'include'
            });

            if (!res.ok) {
                const errText = await res.text();
                throw new Error(`Parte ${i + 1}/${totalChunks} falló: ${errText}`);
            }

            const data = await res.json();
            if (data.dest) {
                finalDest = data.dest;
            }
        }

        if (!finalDest) {
            throw new Error('No se recibió la confirmación del archivo ensamblado.');
        }

        hideToastStatus(false, fileName);
        return finalDest;
    }

    // Interceptar llamadas a window.fetch para redirigir subidas grandes
    window.fetch = async function(resource, init) {
        let url = '';
        if (typeof resource === 'string') {
            url = resource;
        } else if (resource && resource.url) {
            url = resource.url;
        }

        const isGradioUpload = (
            url.includes('/upload') && 
            !url.includes('/upload_chunk') && 
            init && 
            init.method === 'POST' && 
            init.body instanceof FormData
        );

        if (isGradioUpload) {
            const formData = init.body;
            const files = formData.getAll('files');
            const hasLargeFile = files.some(f => (f instanceof Blob) && f.size > CHUNK_SIZE);

            if (hasLargeFile) {
                try {
                    const uploadedPaths = [];
                    for (let fIdx = 0; fIdx < files.length; fIdx++) {
                        const file = files[fIdx];
                        if (file instanceof Blob) {
                            const path = await uploadSingleFileInChunks(file);
                            uploadedPaths.push(path);
                        }
                    }
                    return new Response(JSON.stringify(uploadedPaths), {
                        status: 200,
                        headers: { 'Content-Type': 'application/json' }
                    });
                } catch (err) {
                    console.error('Error en subida por partes:', err);
                    hideToastStatus(true, err.message);
                    return new Response(JSON.stringify({ error: err.message || 'Error en subida' }), {
                        status: 500,
                        headers: { 'Content-Type': 'application/json' }
                    });
                }
            }
        }

        return originalFetch.apply(this, arguments);
    };
})();
</script>
"""

def setup_chunked_upload(demo: gr.Blocks, app: FastAPI, output_dir: str = None):
    """Registra el endpoint de subida en partes en la aplicación FastAPI."""
    @app.post("/upload_chunk")
    async def upload_chunk(
        upload_id: str = Form(...),
        chunk_index: int = Form(...),
        total_chunks: int = Form(...),
        filename: str = Form(...),
        chunk: UploadFile = FastApiFile(...)
    ):
        try:
            cleanup_stale_chunks()
            upload_dir = CHUNK_TEMP_DIR / upload_id
            upload_dir.mkdir(parents=True, exist_ok=True)
            chunk_path = upload_dir / f"chunk_{chunk_index}"

            with open(chunk_path, "wb") as f:
                while True:
                    data = await chunk.read(1024 * 1024)
                    if not data:
                        break
                    f.write(data)

            if chunk_index == total_chunks - 1:
                # Comprobar que estén todas las partes
                for i in range(total_chunks):
                    p = upload_dir / f"chunk_{i}"
                    if not p.exists():
                        return JSONResponse({"error": f"Falta la parte {i+1} en el servidor"}, status_code=400)

                clean_name = client_utils.strip_invalid_filename_characters(filename) or "archivo"
                temp_assembled = upload_dir / "assembled"
                sha256 = hashlib.sha256()

                with open(temp_assembled, "wb") as out_f:
                    for i in range(total_chunks):
                        p = upload_dir / f"chunk_{i}"
                        with open(p, "rb") as in_f:
                            while True:
                                buf = in_f.read(1024 * 1024)
                                if not buf:
                                    break
                                sha256.update(buf)
                                out_f.write(buf)

                # Directorio estándar de Gradio (/tmp/gradio/<hash>/)
                gradio_upload_dir = Path("/tmp/gradio") / sha256.hexdigest()
                gradio_upload_dir.mkdir(parents=True, exist_ok=True)
                final_dest = str(gradio_upload_dir / clean_name)

                shutil.move(str(temp_assembled), final_dest)
                shutil.rmtree(str(upload_dir), ignore_errors=True)

                # Registrar en Gradio para que la validación de archivos temporales lo acepte
                demo.upload_file_set.add(final_dest)

                return {"dest": final_dest}

            return {"status": "chunk_received", "chunk_index": chunk_index}

        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

def launch_server(demo: gr.Blocks, port: int, allowed_paths: list = None, theme=None):
    """Monta Gradio sobre FastAPI con soporte para chunked uploads y ejecuta uvicorn."""
    app = FastAPI()
    setup_chunked_upload(demo, app)
    mounted_app = gr.mount_gradio_app(
        app,
        demo,
        path="/",
        allowed_paths=allowed_paths,
        head=HEAD_HTML,
        theme=theme,
    )
    uvicorn.run(mounted_app, host="0.0.0.0", port=port)
