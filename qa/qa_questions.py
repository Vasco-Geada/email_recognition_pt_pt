"""
qa_questions.py
===============

Definição estruturada de perguntas para o módulo de Question Answering.

Este módulo define um conjunto fixo de perguntas em português europeu
para extrair informações sobre reuniões em emails académicos informais.

Cada pergunta é mapeada para uma categoria semântica e pode ter variações
para aumentar a robustez do modelo.

Project: Email Recognition PT-PT
Version: 1.0
"""

from typing import Dict, List
from dataclasses import dataclass
from enum import Enum


class QuestionCategory(Enum):
    """Categorias de perguntas para QA."""
    PARTICIPANTS = "participants"
    TIME = "time"
    LOCATION = "location"
    TOPIC = "topic"


@dataclass
class Question:
    """Representa uma pergunta estruturada para QA."""
    
    category: QuestionCategory
    primary: str
    variations: List[str]
    expected_answer_type: str
    examples_correct: List[str]
    examples_incorrect: List[str]
    
    def __repr__(self) -> str:
        return f"Question({self.category.value}: {self.primary})"


class QAQuestions:
    """
    Gestor centralizado de perguntas para Question Answering.
    
    Fornece:
    - Perguntas primárias e variações
    - Mapear categoria -> pergunta
    - Validação de respostas
    - Exemplos para treino/teste
    
    Todas as perguntas são em português europeu informal.
    """
    
    # Pergunta sobre PARTICIPANTES
    PARTICIPANTS_PRIMARY = "Quem participa na reunião?"
    PARTICIPANTS_VARIATIONS = [
        "Quem são as pessoas envolvidas?",
        "Com quem é a reunião?",
        "Quem vai estar presente?",
        "Que pessoas vão participar?",
        "Quem está envolvido?",
        "Quem mais vai estar lá?",
        "Há alguém mais na reunião?",
    ]
    PARTICIPANTS_EXAMPLES_CORRECT = [
        "Ana",
        "João",
        "Professor Silva",
        "Rita e João",
        "o meu colega",
        "o orientador",
        " a professora",
        "eu",
        "nós",
        "a equipa",
        "os alunos",
        "a turma",
    ]
    PARTICIPANTS_EXAMPLES_INCORRECT = [
        "reunião",
        "sexta",
        "Teams",
        "10h",
        "na sala",
        "o dataset",
        "a",
        "m",
        "amanhã",
        "hoje",
        "32",
        "cento e vinte e cinco"
    ]
    
    # Pergunta sobre HORA
    TIME_PRIMARY = "Quando é a reunião?"
    TIME_VARIATIONS = [
        "A que horas é a reunião?",
        "Qual é a hora da reunião?",
        "Em que dia e hora?",
        "Quando nos reunimos?",
        "Qual é o dia?",
        "Qual é o horário?",
        "Em que altura?",
        "Para que horas?",
        "A que hora marcou?",
        "Para que horas ficou marcada a reunião?"
        "Para quando é que ficou marcada?"
    ]
    TIME_EXAMPLES_CORRECT = [
        "sexta às 15h",
        "amanhã de manhã",
        "quinta-feira às 10h",
        "segunda de tarde",
        "mais logo",
        "depois de almoço",
        "15h",
        "às 14h30",
        "próxima segunda",
        "daqui a uma semana",
    ]
    TIME_EXAMPLES_INCORRECT = [
        "Ana",
        "Teams",
        "dataset",
        "na sala",
        "o professor",
        "online",
        "presencial",
        "a"
    ]
    
    # Pergunta sobre LOCAL
    LOCATION_PRIMARY = "Onde é a reunião?"
    LOCATION_VARIATIONS = [
        "Qual é o local da reunião?",
        "Em que sítio é?",
        "Qual é o espaço?",
        "Onde vamos reunir?",
        "Em que local?",
        "Para onde é?",
        "Qual é o endereço?",
        "Que sala?",
        "Presencial ou online?",
    ]
    LOCATION_EXAMPLES_CORRECT = [
        "Teams",
        "Zoom",
        "sala 101",
        "na universidade",
        "online",
        "no escritório",
        "presencial",
        "na sala de reuniões",
        "na cantina",
        "à distância",
        "office",
        "remoto",
    ]
    LOCATION_EXAMPLES_INCORRECT = [
        "Ana",
        "sexta",
        "15h",
        "dataset",
        "professor",
    ]
    
    # Pergunta sobre TÓPICO
    TOPIC_PRIMARY = "Qual é o tópico da reunião?"
    TOPIC_VARIATIONS = [
        "O que vai ser discutido?",
        "Qual é o assunto?",
        "Que se vai tratar?",
        "De que é que vamos falar?",
        "Qual é a temática?",
        "Qual é o tema?",
        "O que precisamos discutir?",
        "Qual é o objetivo?",
        "Que vamos abordar?",
        "vamos falar do quê?",
    ]
    TOPIC_EXAMPLES_CORRECT = [
        "dataset",
        "dissertação",
        "o projeto",
        "correção",
        "metodologia",
        "experimentos",
        "resultados",
        "capítulo 3",
        "deadline",
        "cronograma",
    ]
    TOPIC_EXAMPLES_INCORRECT = [
        "Ana",
        "sexta",
        "15h",
        "Teams",
    ]
    
    # Dicionário de mapeamento
    _QUESTIONS: Dict[QuestionCategory, Question] = {
        QuestionCategory.PARTICIPANTS: Question(
            category=QuestionCategory.PARTICIPANTS,
            primary=PARTICIPANTS_PRIMARY,
            variations=PARTICIPANTS_VARIATIONS,
            expected_answer_type="person|group",
            examples_correct=PARTICIPANTS_EXAMPLES_CORRECT,
            examples_incorrect=PARTICIPANTS_EXAMPLES_INCORRECT,
        ),
        QuestionCategory.TIME: Question(
            category=QuestionCategory.TIME,
            primary=TIME_PRIMARY,
            variations=TIME_VARIATIONS,
            expected_answer_type="date|time|relative_time",
            examples_correct=TIME_EXAMPLES_CORRECT,
            examples_incorrect=TIME_EXAMPLES_INCORRECT,
        ),
        QuestionCategory.LOCATION: Question(
            category=QuestionCategory.LOCATION,
            primary=LOCATION_PRIMARY,
            variations=LOCATION_VARIATIONS,
            expected_answer_type="place|room|platform",
            examples_correct=LOCATION_EXAMPLES_CORRECT,
            examples_incorrect=LOCATION_EXAMPLES_INCORRECT,
        ),
        QuestionCategory.TOPIC: Question(
            category=QuestionCategory.TOPIC,
            primary=TOPIC_PRIMARY,
            variations=TOPIC_VARIATIONS,
            expected_answer_type="subject|topic|theme",
            examples_correct=TOPIC_EXAMPLES_CORRECT,
            examples_incorrect=TOPIC_EXAMPLES_INCORRECT,
        ),
    }
    
    @classmethod
    def get_all_questions(cls) -> Dict[QuestionCategory, Question]:
        """
        Retorna todas as perguntas definidas.
        
        Returns:
            Dicionário mapeando categoria -> Question
        """
        return cls._QUESTIONS.copy()
    
    @classmethod
    def get_question(cls, category: QuestionCategory) -> Question:
        """
        Obtém uma pergunta pela categoria.
        
        Args:
            category: Categoria de pergunta
            
        Returns:
            Objeto Question
            
        Raises:
            ValueError: Se categoria não existir
        """
        if category not in cls._QUESTIONS:
            raise ValueError(f"Categoria desconhecida: {category}")
        return cls._QUESTIONS[category]
    
    @classmethod
    def get_primary_question(cls, category: QuestionCategory) -> str:
        """
        Obtém a pergunta primária de uma categoria.
        
        Args:
            category: Categoria de pergunta
            
        Returns:
            String com pergunta primária
        """
        question = cls.get_question(category)
        return question.primary
    
    @classmethod
    def get_all_primary_questions(cls) -> Dict[str, str]:
        """
        Retorna todas as perguntas primárias.
        
        Returns:
            Dicionário mapeando nome_categoria -> pergunta
            
        Example:
            {
                'participants': 'Quem participa na reunião?',
                'time': 'Quando é a reunião?',
                ...
            }
        """
        return {
            cat.value: question.primary
            for cat, question in cls._QUESTIONS.items()
        }
    
    @classmethod
    def get_random_variation(
        cls,
        category: QuestionCategory,
        include_primary: bool = True,
    ) -> str:
        """
        Obtém uma variação aleatória de uma pergunta.
        
        Args:
            category: Categoria de pergunta
            include_primary: Se True, pode devolver pergunta primária
            
        Returns:
            String com variação da pergunta
        """
        import random
        
        question = cls.get_question(category)
        variations = question.variations.copy()
        
        if include_primary:
            variations.insert(0, question.primary)
        
        return random.choice(variations)
    
    @classmethod
    def validate_answer_type(
        cls,
        answer: str,
        category: QuestionCategory,
    ) -> bool:
        """
        Valida se uma resposta tem tipo apropriado.
        
        NOTA: Esta é uma validação básica. Para validação mais robusta,
        usar NER ou modelos semânticos.
        
        Args:
            answer: Resposta a validar
            category: Categoria esperada
            
        Returns:
            True se resposta parece válida para categoria
        """
        if not answer or len(answer.strip()) == 0:
            return False
        
        question = cls.get_question(category)
        
        # Verificar se resposta é muito similar a exemplos incorretos
        answer_lower = answer.lower().strip()
        for incorrect in question.examples_incorrect:
            if incorrect.lower() in answer_lower:
                return False
        
        return True
    
    @classmethod
    def print_all_questions(cls) -> None:
        """Imprime todas as perguntas de forma legível."""
        for cat, question in cls._QUESTIONS.items():
            print(f"\n[{cat.value.upper()}]")
            print(f"  Primária: {question.primary}")
            print(f"  Variações ({len(question.variations)}):")
            for var in question.variations[:3]:
                print(f"    - {var}")
            if len(question.variations) > 3:
                print(f"    ... e mais {len(question.variations) - 3}")


if __name__ == "__main__":
    """Demonstração das perguntas."""
    QAQuestions.print_all_questions()
    
    print("\n" + "="*60)
    print("Perguntas Primárias:")
    for cat_name, question_text in QAQuestions.get_all_primary_questions().items():
        print(f"  {cat_name}: {question_text}")
