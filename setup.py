"""
Script de instalación y configuración para Agente PDF
Automatiza la configuración del entorno virtual y dependencias
"""

import os
import sys
import subprocess
import platform

def print_header(text):
    """Imprime un encabezado."""
    print("\n" + "="*60)
    print(f"✨ {text}")
    print("="*60 + "\n")

def create_venv():
    """Crea un entorno virtual."""
    print_header("CREAR ENTORNO VIRTUAL")
    
    venv_path = "venv"
    
    if os.path.exists(venv_path):
        print(f"✅ El entorno virtual ya existe en {venv_path}")
        return
    
    print("Creando entorno virtual...")
    subprocess.run([sys.executable, "-m", "venv", venv_path], check=True)
    print(f"✅ Entorno virtual creado en {venv_path}")

def get_activate_command():
    """Retorna el comando para activar el venv según el SO."""
    if platform.system() == "Windows":
        return "venv\\Scripts\\activate"
    else:
        return "source venv/bin/activate"

def install_requirements():
    """Instala las dependencias."""
    print_header("INSTALAR DEPENDENCIAS")
    
    if not os.path.exists("requirements.txt"):
        print("❌ requirements.txt no encontrado")
        return False
    
    pip_cmd = [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]
    print("Instalando paquetes (esto puede tardar varios minutos)...")
    
    result = subprocess.run(pip_cmd)
    
    if result.returncode == 0:
        print("✅ Dependencias instaladas correctamente")
        return True
    else:
        print("❌ Error instalando dependencias")
        return False

def create_env_file():
    """Crea archivo .env si no existe."""
    print_header("CONFIGURAR VARIABLES DE ENTORNO")
    
    if os.path.exists(".env"):
        print("✅ El archivo .env ya existe")
        return
    
    env_content = """# Configuración de Agente PDF
OLLAMA_HOST=http://localhost:11434
PDF_MAX_SIZE_MB=50
CACHE_DIR=./cache
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
"""
    
    with open(".env", "w") as f:
        f.write(env_content)
    
    print("✅ Archivo .env creado")

def create_cache_dir():
    """Crea directorio de caché."""
    print_header("CREAR DIRECTORIO DE CACHÉ")
    
    cache_dir = "cache"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
        print(f"✅ Directorio {cache_dir} creado")
    else:
        print(f"✅ Directorio {cache_dir} ya existe")

def check_ollama():
    """Verifica si Ollama está instalado."""
    print_header("VERIFICAR OLLAMA")
    
    try:
        result = subprocess.run(["ollama", "--version"], capture_output=True, timeout=5)
        if result.returncode == 0:
            print(f"✅ Ollama instalado: {result.stdout.decode().strip()}")
            return True
    except:
        pass
    
    print("⚠️ Ollama no detectado localmente")
    print("👉 Descárgalo desde: https://ollama.ai")
    print("👉 O configura OLLAMA_HOST en .env\n")
    return False

def print_next_steps(ollama_found):
    """Imprime los próximos pasos."""
    print_header("PRÓXIMOS PASOS")
    
    activate_cmd = get_activate_command()
    
    print(f"1. Activa el entorno virtual:")
    print(f"   {activate_cmd}\n")
    
    if not ollama_found:
        print("2. Instala Ollama desde: https://ollama.ai\n")
        print("3. Inicia Ollama en otra terminal:")
        print("   ollama serve\n")
        print("4. Descarga un modelo (en otra terminal con Ollama ejecutándose):")
        print("   ollama pull kiwi_kiwi/gemma-4-uncensores:e4b\n")
        print("5. Corre la aplicación:")
    else:
        print("2. Verifica que Ollama esté ejecutándose:")
        print("   ollama serve\n")
        print("3. Descarga el modelo (si no lo tienes):")
        print("   ollama pull kiwi_kiwi/gemma-4-uncensores:e4b\n")
        print("4. Corre la aplicación:")
    
    print("   streamlit run st.py\n")
    
    print("   O para la API FastAPI:")
    print("   python stn8n.py\n")
    
    print("📚 Documentación: README.md")
    print("🌐 Web: http://localhost:8501")
    print("📡 API: http://localhost:8000/docs\n")

def main():
    """Función principal."""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║        🚀 INSTALADOR - AGENTE PDF                            ║
    ║        Lector de PDFs con IA                                ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    try:
        # Ejecutar pasos de instalación
        create_venv()
        install_requirements()
        create_env_file()
        create_cache_dir()
        ollama_found = check_ollama()
        print_next_steps(ollama_found)
        
        print("\n✅ ¡Instalación completada!")
        
    except KeyboardInterrupt:
        print("\n⚠️ Instalación cancelada por el usuario")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error durante la instalación: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
