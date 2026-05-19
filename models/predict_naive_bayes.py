# -*- coding: utf-8 -*-
"""
Script de Predição com o Classificador Naive Bayes.

Este script carrega um modelo treinado e realiza predições em novos emails.
Suporta predições individuais, batch e retorna confiança.

Exemplo de Uso:
    python predict_naive_bayes.py --text "Olá, conseguimos reunir amanhã?"
    python predict_naive_bayes.py --file emails.json --batch
"""

import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Union, Optional
import sys

# Adicionar parent ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.naive_bayes_classifier import NaiveBayesEmailClassifier
from models.utils import combine_text_fields, preprocess_text


# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EmailPredictor:
    """
    Classe para realização de predições de emails.
    
    Attributes:
        classifier (NaiveBayesEmailClassifier): Classificador carregado.
        model_path (str): Caminho do modelo.
        vectorizer_path (str): Caminho do vectorizer.
    """
    
    def __init__(self, model_path: str, vectorizer_path: str) -> None:
        """
        Inicializa o preditor.
        
        Args:
            model_path: Caminho do modelo treinado.
            vectorizer_path: Caminho do vectorizer.
        
        Raises:
            FileNotFoundError: Se os ficheiros não existirem.
        """
        logger.info("Carregando modelo...")
        
        self.classifier = NaiveBayesEmailClassifier()
        
        try:
            self.classifier.load(model_path, vectorizer_path)
            self.model_path = model_path
            self.vectorizer_path = vectorizer_path
            logger.info("✓ Modelo carregado com sucesso")
        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {str(e)}")
            raise
    
    def predict_email(
        self,
        email_text: str,
        return_confidence: bool = True,
        top_n: Optional[int] = None
    ) -> Dict:
        """
        Prediz a classe de um email.
        
        Args:
            email_text: Texto do email a classificar.
            return_confidence: Se True, retorna confiança.
            top_n: Se especificado, retorna top-N classes.
        
        Returns:
            Dicionário com predição e metadados.
        """
        if not email_text or not email_text.strip():
            logger.warning("Texto vazio recebido")
            return {
                'prediction': None,
                'confidence': 0.0,
                'error': 'Texto vazio'
            }
        
        try:
            # Realizar predição
            prediction = self.classifier.predict(email_text)
            
            # Obter probabilidades
            probabilities = self.classifier.predict_proba(email_text)
            confidence = probabilities[prediction]
            
            result = {
                'prediction': prediction,
                'confidence': float(confidence),
                'text': email_text[:100] + ('...' if len(email_text) > 100 else '')
            }
            
            # Se solicitado, retornar top-N
            if top_n:
                sorted_probs = sorted(
                    probabilities.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:top_n]
                result['top_predictions'] = [
                    {'class': cls, 'confidence': float(conf)}
                    for cls, conf in sorted_probs
                ]
            
            return result
            
        except Exception as e:
            logger.error(f"Erro na predição: {str(e)}")
            return {
                'prediction': None,
                'confidence': 0.0,
                'error': str(e)
            }
    
    def predict_email_dict(
        self,
        email_dict: Dict,
        return_confidence: bool = True
    ) -> Dict:
        """
        Prediz a classe de um email em formato dicionário.
        
        Args:
            email_dict: Dicionário com campos 'subject' e 'body'.
            return_confidence: Se True, retorna confiança.
        
        Returns:
            Dicionário com predição.
        """
        try:
            # Combinar campos
            text = combine_text_fields(email_dict)
            
            if not text:
                return {
                    'prediction': None,
                    'confidence': 0.0,
                    'error': 'Email vazio'
                }
            
            return self.predict_email(
                text,
                return_confidence=return_confidence
            )
            
        except Exception as e:
            logger.error(f"Erro ao processar email dict: {str(e)}")
            return {
                'prediction': None,
                'confidence': 0.0,
                'error': str(e)
            }
    
    def predict_batch(
        self,
        texts: List[str],
        return_confidence: bool = True,
        show_progress: bool = True
    ) -> List[Dict]:
        """
        Prediz múltiplos emails.
        
        Args:
            texts: Lista de textos a classificar.
            return_confidence: Se True, retorna confiança.
            show_progress: Se True, mostra progresso.
        
        Returns:
            Lista de dicionários com predições.
        """
        results = []
        
        total = len(texts)
        for idx, text in enumerate(texts, 1):
            if show_progress:
                logger.info(f"Processando {idx}/{total}...")
            
            result = self.predict_email(
                text,
                return_confidence=return_confidence
            )
            results.append(result)
        
        return results
    
    def predict_from_json(
        self,
        json_path: str,
        return_confidence: bool = True
    ) -> List[Dict]:
        """
        Prediz emails a partir de ficheiro JSON.
        
        Args:
            json_path: Caminho do ficheiro JSON.
            return_confidence: Se True, retorna confiança.
        
        Returns:
            Lista de dicionários com predições.
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                emails = json.load(f)
            
            if not isinstance(emails, list):
                emails = [emails]
            
            logger.info(f"Carregados {len(emails)} emails de {json_path}")
            
            results = []
            for idx, email in enumerate(emails, 1):
                logger.info(f"Processando email {idx}/{len(emails)}...")
                result = self.predict_email_dict(
                    email,
                    return_confidence=return_confidence
                )
                # Adicionar ID se existir
                if 'email_id' in email:
                    result['email_id'] = email['email_id']
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Erro ao ler JSON: {str(e)}")
            raise


def print_result(result: Dict, detailed: bool = False) -> None:
    """
    Imprime resultado de predição de forma formatada.
    
    Args:
        result: Dicionário com resultado.
        detailed: Se True, mostra informações completas.
    """
    print("\n" + "=" * 70)
    
    if result.get('error'):
        print(f"❌ Erro: {result['error']}")
    else:
        print(f"📧 Predição: {result['prediction']}")
        print(f"📊 Confiança: {result['confidence']:.2%}")
        
        if detailed and 'top_predictions' in result:
            print("\n📈 Top Predições:")
            for pred in result['top_predictions']:
                print(f"   - {pred['class']}: {pred['confidence']:.2%}")
        
        if 'text' in result and detailed:
            print(f"\n💬 Texto: {result['text']}")
    
    print("=" * 70 + "\n")


def main():
    """Função principal."""
    parser = argparse.ArgumentParser(
        description="Predição de classes de emails com Naive Bayes"
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='models/naive_bayes_model.joblib',
        help='Caminho do modelo treinado'
    )
    parser.add_argument(
        '--vectorizer',
        type=str,
        default='models/naive_bayes_vectorizer.joblib',
        help='Caminho do vectorizer'
    )
    
    # Modos de input
    parser.add_argument(
        '--text',
        type=str,
        help='Texto a classificar'
    )
    parser.add_argument(
        '--file',
        type=str,
        help='Ficheiro JSON com emails'
    )
    parser.add_argument(
        '--batch',
        action='store_true',
        help='Processar ficheiro em batch'
    )
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Modo interativo'
    )
    
    # Opções
    parser.add_argument(
        '--top-n',
        type=int,
        default=3,
        help='Mostrar top-N predições'
    )
    parser.add_argument(
        '--detailed',
        action='store_true',
        help='Mostra informações detalhadas'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Guardar resultados em ficheiro JSON'
    )
    
    args = parser.parse_args()
    
    # Validar caminhos do modelo
    if not Path(args.model).exists():
        logger.error(f"Modelo não encontrado: {args.model}")
        sys.exit(1)
    
    if not Path(args.vectorizer).exists():
        logger.error(f"Vectorizer não encontrado: {args.vectorizer}")
        sys.exit(1)
    
    # Carregar preditor
    predictor = EmailPredictor(args.model, args.vectorizer)
    
    results = []
    
    # Modo com texto
    if args.text:
        logger.info(f"Classificando texto...")
        result = predictor.predict_email(
            args.text,
            top_n=args.top_n
        )
        results.append(result)
        print_result(result, detailed=args.detailed)
    
    # Modo com ficheiro
    elif args.file:
        if not Path(args.file).exists():
            logger.error(f"Ficheiro não encontrado: {args.file}")
            sys.exit(1)
        
        try:
            results = predictor.predict_from_json(args.file)
            
            for idx, result in enumerate(results, 1):
                print(f"\n{'='*70}")
                print(f"Email {idx}/{len(results)}")
                print_result(result, detailed=args.detailed)
            
        except Exception as e:
            logger.error(f"Erro ao processar ficheiro: {str(e)}")
            sys.exit(1)
    
    # Modo interativo
    elif args.interactive:
        logger.info("Modo interativo iniciado (tipo 'sair' para terminar)")
        while True:
            try:
                text = input("\n📧 Digite o texto do email (ou 'sair'): ").strip()
                if text.lower() == 'sair':
                    break
                
                result = predictor.predict_email(
                    text,
                    top_n=args.top_n
                )
                results.append(result)
                print_result(result, detailed=args.detailed)
                
            except KeyboardInterrupt:
                print("\n\nEncerrando...")
                break
            except Exception as e:
                logger.error(f"Erro: {str(e)}")
    
    else:
        parser.print_help()
        return
    
    # Guardar resultados
    if args.output and results:
        try:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logger.info(f"✓ Resultados guardados em: {args.output}")
        except Exception as e:
            logger.error(f"Erro ao guardar resultados: {str(e)}")


if __name__ == '__main__':
    main()
