# Email Anonymization Module

Hybrid anonymization for Portuguese academic emails before NLP processing.

Recommended flow:

```text
raw email
  -> anonymization
  -> preprocessing
  -> intent classification
  -> trigger extraction
  -> argument extraction
  -> temporal normalization
```

## Files

- `anonymization.py`: main orchestrator and public functions.
- `regex_anonymizer.py`: emails, phone numbers, URLs and academic identifiers.
- `ner_anonymizer.py`: spaCy NER plus academic person-name heuristics.
- `anonymization_config.py`: whitelists, academic institutions and placeholders.
- `test_anonymization.py`: required tests and regression examples.

## Usage

```python
from preprocessing.anonymization import anonymize_email

email = {
    "subject": "Reunião com Ana",
    "body": "Boas Ana, podemos reunir amanhã no Teams?",
    "label": "agendamento_reuniao",
}

result = anonymize_email(email)
print(result["subject"])
print(result["body"])
```

Output:

```json
{
  "subject": "Reunião com [PESSOA_1]",
  "body": "Boas [PESSOA_1], podemos reunir amanhã no Teams?",
  "label": "agendamento_reuniao",
  "anonymization": {
    "entities": [
      {
        "replacement": "[PESSOA_1]",
        "type": "PERSON",
        "start": 12,
        "end": 15,
        "method": "NAME_LEXICON",
        "field": "subject"
      }
    ],
    "mode": "anonymize"
  }
}
```

For controlled review, use pseudonymization mode:

```python
result = anonymize_email(email, keep_mapping=True)
```

This includes original values and a mapping. Do not export this version to a final dataset unless access is controlled.

## What Is Anonymized

- Person names.
- Professor/student names in academic contexts.
- Organizations and locations detected by Portuguese NER.
- Portuguese universities and institutions from curated lists.
- Email addresses.
- Portuguese phone numbers.
- Student numbers and academic identifiers.
- URLs.

## What Is Preserved

The platform whitelist is intentionally preserved because these tokens are useful for location/channel extraction:

- Teams
- Zoom
- Discord
- Moodle
- GitHub
- Google Meet
- Outlook
- Gmail
- Slack
- Trello
- Notion

Temporal expressions are not targeted by the anonymizer, so strings such as `amanhã`, `sexta às 15h` and `depois de almoço` should remain available for temporal normalization.

## Testing

```bash
python preprocessing/test_anonymization.py
```

The tests cover:

- simple names;
- repeated names with consistent placeholders;
- emails;
- phone numbers;
- universities;
- platform whitelist;
- mixed informal academic emails;
- subject/body consistency.

## Technical Notes

The module combines:

- regex for high-precision patterns;
- spaCy `pt_core_news_sm` for NER;
- curated academic entity lists;
- conservative name lexicons and academic-context rules.

Overlapping entities are resolved by priority and span length. Replacements are applied from right to left, preserving offsets from the original text.

## Limitations

- NER may miss informal names, nicknames or abbreviated signatures.
- The name lexicon can create false positives for words that are also surnames.
- Organization/location NER is not perfect in short emails.
- Some personal URLs may be hard to distinguish from institutional or platform URLs.
- True anonymization is hard to guarantee with free text because rare contextual clues can still identify people.

## RGPD Notes

Anonymization and pseudonymization are different:

- anonymization removes the practical ability to recover the original identity;
- pseudonymization replaces identifiers but keeps a mapping, so it is still personal data if the mapping exists.

For a dissertation dataset, keep only the anonymized output by default. Use `keep_mapping=True` only for restricted manual validation, then discard or protect mappings separately.

Recommended validation:

- manually inspect a random sample of anonymized emails;
- inspect all emails with zero detected entities but likely personal content;
- inspect all emails with many replacements to catch over-anonymization;
- maintain a small error log for false positives and false negatives.

