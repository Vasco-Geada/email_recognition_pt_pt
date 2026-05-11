#!/usr/bin/env python3
"""
setup.py
========

Script de setup para o sistema de Gold Annotations.

Uso:
    python setup.py --help
    python setup.py install
    python setup.py verify
    python setup.py demo

Author: Generated for Email Recognition PT-PT Project
"""

import sys
import json
import argparse
from pathlib import Path

# Cores para terminal (com fallback para Windows)
class Colors:
    GREEN = '\033[92m' if sys.stdout.isatty() else ''
    RED = '\033[91m' if sys.stdout.isatty() else ''
    YELLOW = '\033[93m' if sys.stdout.isatty() else ''
    BLUE = '\033[94m' if sys.stdout.isatty() else ''
    RESET = '\033[0m' if sys.stdout.isatty() else ''


def print_header(text):
    """Imprime header formatado"""
    print(f"\n{Colors.BLUE}{'='*70}")
    print(f"{text}")
    print(f"{'='*70}{Colors.RESET}\n")


def print_success(text):
    """Imprime mensagem de sucesso"""
    print(f"{Colors.GREEN}[OK]{Colors.RESET} {text}")


def print_error(text):
    """Imprime mensagem de erro"""
    print(f"{Colors.RED}[ERROR]{Colors.RESET} {text}")


def print_warning(text):
    """Imprime mensagem de aviso"""
    print(f"{Colors.YELLOW}[WARN]{Colors.RESET} {text}")


def print_info(text):
    """Imprime mensagem informativa"""
    print(f"{Colors.BLUE}[INFO]{Colors.RESET} {text}")


def verify_python_version():
    """Verifica versão de Python"""
    print_info("Verificando versão de Python...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 11):
        print_error(f"Python 3.11+ necessário (actual: {version.major}.{version.minor})")
        return False
    
    print_success(f"Python {version.major}.{version.minor}.{version.micro} OK")
    return True


def verify_modules():
    """Verifica módulos necessários"""
    print_info("Verificando módulos...")
    
    required_modules = [
        'json', 're', 'pathlib', 'dataclasses', 'typing', 'collections'
    ]
    
    for module in required_modules:
        try:
            __import__(module)
            print_success(f"Módulo '{module}' disponível")
        except ImportError:
            print_error(f"Módulo '{module}' não disponível")
            return False
    
    return True


def verify_files():
    """Verifica ficheiros do projeto"""
    print_info("Verificando ficheiros...")
    
    required_files = [
        'heuristic_extractors.py',
        'validators.py',
        'gold_annotations_generator.py',
        'evaluate_annotations.py',
        '__init__.py',
        'config.py',
        'example_usage.py',
        'test_gold_annotations.py',
        'README.md',
    ]
    
    base_dir = Path(__file__).parent
    all_found = True
    
    for filename in required_files:
        filepath = base_dir / filename
        if filepath.exists():
            print_success(f"Ficheiro '{filename}' encontrado")
        else:
            print_error(f"Ficheiro '{filename}' NÃO encontrado")
            all_found = False
    
    return all_found


def verify_output_dir():
    """Verifica/cria diretório de output"""
    print_info("Verificando diretório de output...")
    
    output_dir = Path(__file__).parent / 'output'
    try:
        output_dir.mkdir(exist_ok=True)
        print_success(f"Diretório de output criado: {output_dir}")
        return True
    except Exception as e:
        print_error(f"Falha ao criar diretório: {e}")
        return False


def test_imports():
    """Testa imports dos módulos principais"""
    print_info("Testando imports...")
    
    try:
        from heuristic_extractors import HeuristicAnnotationExtractor
        print_success("HeuristicAnnotationExtractor importado")
        
        from validators import AnnotationValidator
        print_success("AnnotationValidator importado")
        
        from gold_annotations_generator import GoldAnnotationsGenerator
        print_success("GoldAnnotationsGenerator importado")
        
        from evaluate_annotations import AnnotationEvaluator
        print_success("AnnotationEvaluator importado")
        
        return True
    except ImportError as e:
        print_error(f"Erro ao importar: {e}")
        return False


def run_tests():
    """Executa testes unitários"""
    print_info("Executando testes...")
    
    try:
        import test_gold_annotations
        # Tests rodão automaticamente
        print_success("Testes executados")
        return True
    except Exception as e:
        print_error(f"Erro ao executar testes: {e}")
        return False


def create_sample_project():
    """Cria projeto de amostra"""
    print_info("Criando projeto de amostra...")
    
    base_dir = Path(__file__).parent
    sample_dir = base_dir / 'samples'
    
    try:
        sample_dir.mkdir(exist_ok=True)
        
        # Criar sample emails
        sample_emails = [
            {
                "subject": "Reunião amanhã",
                "body": "Boas Ana, podemos reunir amanhã às 15h no Teams?",
                "label": "agendamento_reuniao"
            },
            {
                "subject": "Cancelar",
                "body": "Não consigo aparecer sexta. Desculpa!",
                "label": "cancelamento_reuniao"
            },
            {
                "subject": "Confirmado",
                "body": "Vejo te sexta às 14h.",
                "label": "reuniao_confirmada"
            },
        ]
        
        sample_file = sample_dir / 'sample_emails.json'
        with open(sample_file, 'w', encoding='utf-8') as f:
            json.dump(sample_emails, f, indent=2, ensure_ascii=False)
        
        print_success(f"Projeto de amostra criado em: {sample_dir}")
        return True
    except Exception as e:
        print_error(f"Erro ao criar amostra: {e}")
        return False


def verify_installation():
    """Verifica instalação completa"""
    print_header("VERIFICAÇÃO DE INSTALAÇÃO")
    
    checks = [
        ("Versão Python", verify_python_version),
        ("Módulos", verify_modules),
        ("Ficheiros", verify_files),
        ("Diretório Output", verify_output_dir),
        ("Imports", test_imports),
    ]
    
    results = {}
    for name, check_func in checks:
        print(f"\n{Colors.BLUE}Verificando {name}...{Colors.RESET}")
        results[name] = check_func()
    
    print_header("RESUMO DE VERIFICAÇÃO")
    
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    
    print(f"Verificações: {passed}/{total} passadas")
    
    if passed == total:
        print_success("Sistema pronto para uso!")
        return True
    else:
        print_warning("Algumas verificações falharam")
        return False


def show_usage():
    """Mostra instruções de uso"""
    print_header("INSTRUÇÕES DE USO")
    
    print("""
1. Gerar Gold Annotations:

   python gold_annotations_generator.py input.json output.json
   
   Onde:
   - input.json: Ficheiro JSON com emails
   - output.json: Ficheiro JSON com anotações

2. Validar Anotações:

   python -c "
   from validators import AnnotationValidator
   import json
   
   with open('output.json') as f:
       anns = json.load(f)
   
   validator = AnnotationValidator()
   result = validator.validate_batch(anns)
   print(f'Válido: {result.is_valid}')
   "

3. Avaliar Predições:

   python evaluate_annotations.py gold.json predictions.json -o report.json

4. Executar Exemplos:

   python example_usage.py

5. Executar Testes:

   python test_gold_annotations.py

Documentação:
- README.md: Visão geral
- QUICKSTART.md: Quick start (5 minutos)
- TECHNICAL_DOCUMENTATION.md: Documentação técnica
    """)


def main():
    """Função principal"""
    parser = argparse.ArgumentParser(
        description='Setup para Sistema de Gold Annotations'
    )
    parser.add_argument(
        'action',
        nargs='?',
        default='verify',
        choices=['verify', 'install', 'test', 'demo', 'usage'],
        help='Ação a executar'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Modo verbose'
    )
    
    args = parser.parse_args()
    
    print_header("SISTEMA DE GOLD ANNOTATIONS - SETUP")
    
    if args.action == 'verify':
        success = verify_installation()
        sys.exit(0 if success else 1)
    
    elif args.action == 'install':
        print_info("Instalando...")
        success = (verify_installation() and 
                  verify_output_dir() and 
                  create_sample_project())
        
        if success:
            print_success("Instalação concluída com sucesso!")
            print_info("Execute: python example_usage.py")
        sys.exit(0 if success else 1)
    
    elif args.action == 'test':
        print_info("Executando testes...")
        import subprocess
        result = subprocess.run([sys.executable, 'test_gold_annotations.py'])
        sys.exit(result.returncode)
    
    elif args.action == 'demo':
        print_info("Executando demonstração...")
        import subprocess
        result = subprocess.run([sys.executable, 'example_usage.py'])
        sys.exit(result.returncode)
    
    elif args.action == 'usage':
        show_usage()
        sys.exit(0)


if __name__ == '__main__':
    main()
