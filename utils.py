"""
Utilidades compartidas para el proyecto Agente PDF v2
- Validación de PDFs
- Estadísticas de caché
- Funciones auxiliares
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, Tuple


# Configuración
CACHE_DIR = Path("./cache")
PDF_MAX_SIZE_MB = 50

# Crear directorio de caché si no existe
CACHE_DIR.mkdir(exist_ok=True)


def validate_pdf(file_content: bytes, filename: str) -> Tuple[bool, str]:
    """
    Valida un archivo PDF.
    
    Args:
        file_content: Contenido en bytes del archivo
        filename: Nombre del archivo
        
    Returns:
        (es_valido, mensaje)
    """
    # Validar extensión
    if not filename.lower().endswith('.pdf'):
        return False, "❌ El archivo debe ser un PDF"
    
    # Validar tamaño
    size_mb = len(file_content) / (1024 * 1024)
    if size_mb > PDF_MAX_SIZE_MB:
        return False, f"❌ El PDF es demasiado grande ({size_mb:.1f}MB). Máximo: {PDF_MAX_SIZE_MB}MB"
    
    # Validar que tenga contenido
    if len(file_content) < 100:
        return False, "❌ El archivo parece estar vacío"
    
    # Validar que sea PDF (magic number)
    if not file_content.startswith(b'%PDF'):
        return False, "❌ El archivo no parece ser un PDF válido"
    
    return True, "✅ PDF válido"


def get_cache_stats() -> Dict:
    """Obtiene estadísticas del caché."""
    if not CACHE_DIR.exists():
        return {
            "total_files": 0,
            "cache_size": 0,
            "cache_size_mb": 0,
            "hits": 0
        }
    
    # Contar archivos
    files = list(CACHE_DIR.glob("*"))
    total_files = len(files)
    
    # Calcular tamaño
    total_size = sum(f.stat().st_size for f in files if f.is_file())
    total_size_mb = round(total_size / (1024 * 1024), 2)
    
    return {
        "total_files": total_files,
        "cache_size": total_size,
        "cache_size_mb": total_size_mb,
        "hits": 0  # Podrías implementar un contador de hits
    }


def clear_cache() -> str:
    """Limpia toda la caché."""
    if CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR)
        CACHE_DIR.mkdir(exist_ok=True)
        return "✅ Caché limpiada correctamente"
    return "ℹ️ La caché ya estaba vacía"


def get_cache_info() -> Dict:
    """Alias para get_cache_stats (compatibilidad)."""
    stats = get_cache_stats()
    return {
        "cached_files": stats["total_files"],
        "cache_size_mb": stats["cache_size_mb"],
        "cache_dir": str(CACHE_DIR.absolute())
    }

