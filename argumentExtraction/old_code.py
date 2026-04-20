TEST_EMAILS = [
        {
            "id": "test_001",
            "subject": "Reunião - Projeto de IA",
            "body": """
            Gostaria de agendar uma reunião com o Dr. Silva e Maria Costa para discutir 
            o projeto de IA. Podemos fazer amanhã à tarde na sala 203? 
            
            Caso não seja possível, próxima segunda de manhã também fica bem.
            
            Obrigado,
            João
            """,
            "intent": "agendamento_reuniao",
            "trigger": "agendar",
            "expected": {
                "participants": ["Dr. Silva", "Maria Costa"],
                "times": ["amanhã", "à tarde", "próxima segunda", "de manhã"],
                "locations": ["sala 203"],
                "topics": ["projeto de IA"],
            }
        },
        {
            "id": "test_002",
            "subject": "Cancelamento - Reunião sexta",
            "body": """
            Infelizmente não posso comparecer à reunião de sexta às 15h na sala 102.
            
            Podemos reagendar para a próxima semana?
            
            Desculpa,
            Paulo
            """,
            "intent": "cancelamento_reuniao",
            "trigger": "cancelamento",
            "expected": {
                "participants": ["Paulo"],  # Or extract from signature
                "times": ["sexta", "15h", "próxima semana"],
                "locations": ["sala 102"],
                "topics": [],  # No specific topic mentioned
            }
        },
        {
            "id": "test_003",
            "subject": "Confirmação - Reunião com o Client",
            "body": """
            Confirmamos a reunião com a Cliente Maria da Silva para 5 de março de 2024 
            às 10:00 no Auditório A para apresentação do protótipo.
            
            Presentes: João Silva (PM), Pedro Costa (Dev), Ana Pereira (Designer), 
            Maria da Silva (Client), contact@clientcompany.com
            
            Já estão confirmadas: sala com projetor, café, materiais de suporte.
            """,
            "intent": "reuniao_confirmada",
            "trigger": None,
            "expected": {
                "participants": ["Maria da Silva", "João Silva", "Pedro Costa", "Ana Pereira", "contact@clientcompany.com"],
                "times": ["5 de março de 2024", "10:00"],
                "locations": ["Auditório A"],
                "topics": ["apresentação do protótipo"],
            }
        },
        {
            "id": "test_004",
            "subject": "Reunião informal - Almoço com equipa",
            "body": """
            Depois de almoço temos reunião da equipa no 1º andar, sala 205?
            
            Podemos discutir o cronograma do Q4 e recursos necessários.
            
            Confirma?
            """,
            "intent": "agendamento_reuniao",
            "trigger": None,
            "expected": {
                "participants": [],  # Generic "equipa"
                "times": ["Depois de almoço"],
                "locations": ["1º andar", "sala 205"],
                "topics": ["cronograma do Q4", "recursos"],
            }
        },
        {
            "id": "test_005",
            "subject": "Dágil - Retrospectiva da semana",
            "body": """
            Reunião de retrospectiva amanhã de manhã, às 9h na sala 301.
            
            Agenda: resultados da sprint, issues de qualidade, próximos passos.
            
            Participantes: João, Paulo, Ana, Bruno.
            Email: retrospective@team.com
            """,
            "intent": "agendamento_reuniao",
            "trigger": None,
            "expected": {
                "participants": ["João", "Paulo", "Ana", "Bruno", "retrospective@team.com"],
                "times": ["amanhã", "de manhã", "9h"],
                "locations": ["sala 301"],
                "topics": ["retrospectiva", "sprint", "qualidade"],
            }
        },
    ]
    