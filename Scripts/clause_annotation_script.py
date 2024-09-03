import nltk
from nltk.stem import WordNetLemmatizer
from datasets import load_dataset
import json

# Ensure you have the necessary NLTK data
nltk.download('wordnet')

def contains_symbols(sentence):
    # Check if both `` and '' are in the sentence
    return any('``' in word for word, _ in sentence) and any("''" in word for word, _ in sentence)

def identify_adverbial_clauses(pos_tags):
    # More exhaustive list of subordinating conjunctions
    subordinating_conjunctions = {
        'because', 'if', 'when', 'since', 'although', 'while', 'after', 'before',
        'though', 'unless', 'until', 'even though', 'as if', 'as though', 'so that',
        'in order to', 'provided that', 'as long as', 'once', 'now that', 'as'
    }
    punctuation = {',', '.', ';'}

    found_clauses = []

    i = 0
    while i < len(pos_tags):
        word, tag = pos_tags[i]

        # Check if current word is a subordinating conjunction
        if tag == 'IN' or tag == 'WRB':
            potential_conjunction = word.lower()

            # Check for multi-word subordinating conjunctions
            for conj in subordinating_conjunctions:
                conj_words = conj.split()
                if len(conj_words) > 1 and i + len(conj_words) - 1 < len(pos_tags):
                    if all(pos_tags[i + k][0].lower() == conj_words[k] for k in range(len(conj_words))):
                        potential_conjunction = conj
                        break
                elif len(conj_words) == 1 and potential_conjunction == conj_words[0]:
                    break

            # Check the next word after the conjunction for edge cases
            next_word_index = i + len(potential_conjunction.split())
            if next_word_index < len(pos_tags):
                next_tag = pos_tags[next_word_index][1]

                # Skip if next word is an IN (e.g., 'because of') or VBG (e.g., 'after doing')
                if next_tag == 'IN' or next_tag == 'VBG' or next_tag == 'JJ':
                    i += len(potential_conjunction.split())
                    continue

                # If it's a valid subordinating conjunction, process the clause
                if potential_conjunction in subordinating_conjunctions:
                    clause_start = i
                    clause_end = None
                    verb_found = False

                    # Continue to the end of the clause or sentence
                    j = next_word_index
                    while j < len(pos_tags) and pos_tags[j][0] not in punctuation:
                        # Check if a verb is present
                        if pos_tags[j][1].startswith('VB'):  # POS tags for verbs start with 'VB'
                            verb_found = True
                        j += 1

                    clause_end = j

                    # Only consider it an adverbial clause if a verb was found
                    if verb_found:
                        clause = pos_tags[clause_start:clause_end]
                        found_clauses.append({'clause': clause})

                    # Move the index to the end of the current clause
                    i = clause_end - 1

        i += 1

    return found_clauses if found_clauses else None  # Return all found clauses or None if no clause found

def identify_noun_clauses(pos_tags):
    # Trigger verbs for object clauses
    object_clause_trigger_verbs = {
        'add', 'agree', 'admit', 'advise', 'affirm', 'allege', 'announce', 'argue', 'assume',
        'assure', 'assert', 'bet', 'boast', 'caution', 'charge', 'cite',
        'concede', 'conclude', 'confess', 'confirm', 'contend', 'convince',
        'decide', 'deduce', 'deny', 'determine', 'doubt', 'estimate',
        'explain', 'fear', 'figure', 'find', 'forget', 'guarantee', 'guess',
        'hear', 'hint', 'hope', 'hypothesize', 'imagine', 'imply', 'indicate', 'inform', 'insist',
        'instruct', 'interpret', 'know', 'learn', 'maintain', 'mention',
        'note', 'notice', 'observe', 'pledge', 'postulate', 'presume',
        'proclaim', 'propose', 'prove', 'realize', 'recall', 'recognize',
        'recollect', 'recommend', 'recount', 'regret', 'remark', 'remember',
        'report', 'request', 'resolve', 'reveal', 'rule', 'say', 'speculate',
        'state', 'stipulate', 'stress', 'suggest', 'suppose', 'surmise',
        'suspect', 'swear', 'teach', 'testify', 'think', 'understand', 'urge',
        'vow', 'warn', 'wish', 'worry'
    }

    # Trigger nouns for appositive clauses
    appositive_words = {
        "assumption", "belief", "claim", "concept", "concern", "conviction",
        "declaration", "demand", "desire", "discovery", "expectation", "fact",
        "fear", "feeling", "hope", "idea", "insight", "issue", "knowledge",
        "notion", "observation", "opinion", "order", "perception", "plan",
        "possibility", "prediction", "principle", "promise", "proposal",
        "realization", "recognition", "requirement", "rule", "statement",
        "suggestion", "suspicion", "theory", "thought", "understanding",
        "warning", "wish", "sign"
    }

    # POS tags for verbs, determiners, adjectives, nouns/pronouns, wh-words
    verbs = {'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'}
    determiners_adjectives = {'DT', 'JJ', 'CD'}
    nouns_pronouns = {'PRP', 'PRP$', 'NN', 'NNS', 'NNP', 'NNPS'}
    wh_words = {'WP', 'WRB', 'WDT'}
    punctuation = {'.', ';'}

    lemmatizer = WordNetLemmatizer()
    found_clauses = []

    for i, (word, tag) in enumerate(pos_tags):
        # Case 1: Detect "that" indicating an object clause
        if word.lower() == 'that' and tag == 'IN':
            # Ensure it is not preceded by an adverb (RB) like in "so that" or "so quickly that"
            if i > 0 and pos_tags[i - 1][1] != 'RB':
                # Check for appositive trigger words
                if i > 1 and pos_tags[i - 1][1] in {'NN', 'NNS'}:
                    base_form = lemmatizer.lemmatize(pos_tags[i - 1][0].lower(), 'n')
                    if base_form in appositive_words:
                        # Detected an appositive clause starting with "that"
                        j = i + 1
                        # Allow for determiners, adjectives before the noun
                        while j < len(pos_tags) and pos_tags[j][1] in determiners_adjectives.union({'IN'}):
                            j += 1
                        # Look for the noun following "that"
                        if j < len(pos_tags) and pos_tags[j][1] in nouns_pronouns:
                            clause_end = j
                            while clause_end + 1 < len(pos_tags) and pos_tags[clause_end + 1][0] not in punctuation:
                                clause_end += 1
                            clause = pos_tags[i:clause_end + 1]
                            found_clauses.append({'clause': clause})
                            continue

                # Check for object clause detection (preceding verb)
                for j in range(i - 1, -1, -1):
                    if pos_tags[j][1].startswith('VB'):
                        base_form = lemmatizer.lemmatize(pos_tags[j][0].lower(), 'v')
                        if base_form in object_clause_trigger_verbs:
                            clause_end = i
                            while clause_end + 1 < len(pos_tags) and pos_tags[clause_end + 1][0] not in punctuation:
                                clause_end += 1
                            clause = pos_tags[i:clause_end + 1]
                            found_clauses.append({'clause': clause})
                            break

        # Case 2: Detect noun clauses without "that"
        if tag.startswith('VB'):
            base_form = lemmatizer.lemmatize(word.lower(), 'v')
            if base_form in object_clause_trigger_verbs:
                # Check for a noun/pronoun before the verb, allowing for an optional comma
                k = i - 1
                while k >= 0 and pos_tags[k][1] == ',':
                    k -= 1  # Skip over commas to check for preceding noun/pronoun
                if k >= 0 and pos_tags[k][1] in nouns_pronouns:
                    j = i + 1
                    if j < len(pos_tags) and pos_tags[j][1] == 'EX':
                        j += 1  # Move past 'there'
                    while j < len(pos_tags) and (pos_tags[j][1] in determiners_adjectives.union(nouns_pronouns) or pos_tags[j][1] == 'MD'):
                        j += 1
                    if j < len(pos_tags) and pos_tags[j][1] in verbs:
                        clause_end = j
                        while clause_end + 1 < len(pos_tags) and pos_tags[clause_end + 1][0] not in punctuation:
                            clause_end += 1
                        clause = pos_tags[i + 1:clause_end + 1]
                        found_clauses.append({'clause': clause})

    return found_clauses if found_clauses else None  # Return all found clauses or None if no clause found

def identify_attributive_clauses(pos_tags):
    # Rule 1: Identify relative pronoun or adverb
    relative_pronouns_adverbs = {'WP', 'WDT', 'WP$', 'WRB'}
    nouns_pronouns = {'PRP', 'PRP$', 'NN', 'NNS', 'NNP', 'NNPS', 'DT'}
    determiners_adjectives_adverbs = {'DT', 'JJ', 'CD', 'RB'}
    verbs = {'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'}
    punctuation = {'.', ',', '#', ';'}

    found_clauses = []

    for i, (word, tag) in enumerate(pos_tags):
        if tag in relative_pronouns_adverbs:
            # Skip if the word is "what"
            if word.lower() == 'what':
                continue

            # Found a relative pronoun or adverb (Rule 1)
            clause_start = i
            clause_end = None
            
            # Rule 2: Allow for optional noun/pronoun, determiner, adjective, adverb (RB), and punctuation like commas and #
            j = i + 1
            while j < len(pos_tags) and (pos_tags[j][1] in nouns_pronouns.union(determiners_adjectives_adverbs) or pos_tags[j][1] == 'MD' or pos_tags[j][1] in {',', '#'}):
                j += 1

            # Rule 3: Look for the main verb of the clause
            if j < len(pos_tags) and pos_tags[j][1] in verbs:
                clause_end = j
                # Continue capturing if the next word is also a verb (greedy capture)
                while clause_end + 1 < len(pos_tags) and (pos_tags[clause_end + 1][1] in verbs.union(determiners_adjectives_adverbs).union(nouns_pronouns).union({',', '#'})):
                    clause_end += 1
            else:
                # If no verb is found, set clause_end to the next natural ending point
                clause_end = j
                while clause_end < len(pos_tags) and pos_tags[clause_end][0] not in punctuation:
                    clause_end += 1

            # Rule 4: Identify antecedent noun or pronoun (immediately before the relative pronoun/adverb)
            antecedent_index = clause_start - 1
            if antecedent_index >= 0 and pos_tags[antecedent_index][0] in punctuation:
                antecedent_index -= 1  # Skip over a comma or #

            if antecedent_index >= 0 and pos_tags[antecedent_index][1] in nouns_pronouns:
                antecedent = pos_tags[antecedent_index]

                # Extract the attributive clause
                clause = pos_tags[clause_start:clause_end + 1]
                found_clauses.append({
                    'antecedent': antecedent,
                    'clause': clause
                })

    return found_clauses if found_clauses else None  # Return all found clauses or None if none found

sa_ds = load_dataset("batterydata/pos_tagging")
# print(sa_ds)

sample_size = 400
target_size = 120
train_set = sa_ds["train"]
# print(list(zip(train_set["words"][:sample_size], train_set["labels"][:sample_size])))
pos_tagged_sentences = [list(zip(words, labels)) for words, labels in zip(train_set["words"][:sample_size], train_set["labels"][:sample_size])]

effective_sentences = []  # Used to store sentences that have at least one type of clause

# Loop through the sentences and apply the functions
for i, pos_tags in enumerate(pos_tagged_sentences, start=1):
    if len(effective_sentences) >= target_size:
        break

    # Exclude sentences that contain the symbols
    if contains_symbols(pos_tags):
        continue

    attributive_clause_results = identify_attributive_clauses(pos_tags)
    noun_clause_results = identify_noun_clauses(pos_tags)
    adverbial_clause_results = identify_adverbial_clauses(pos_tags)
    sentence = " ".join([word for word, _ in pos_tags])

    # print(f"Sentence {i}: {sentence}")

    clauses_dict = {
        'index': "TBC",
        'sentence': sentence,
        'attributive_clauses': [' '.join([word for word, _ in result['clause']]) for result in attributive_clause_results] if attributive_clause_results else [],
        'noun_clauses': [' '.join([word for word, _ in result['clause']]) for result in noun_clause_results] if noun_clause_results else [],
        'adverbial_clauses': [' '.join([word for word, _ in result['clause']]) for result in adverbial_clause_results] if adverbial_clause_results else []
    }


    if clauses_dict['attributive_clauses'] or clauses_dict['noun_clauses'] or clauses_dict['adverbial_clauses']:
        effective_sentences.append(clauses_dict)

for i, entry in enumerate(effective_sentences, start=1):
    print(f"Sentence {i}: {entry['sentence']}")
    
    if entry['attributive_clauses']:
        print("  Attributive Clauses:")
        for clause in entry['attributive_clauses']:
            print(f"    - {clause}")
    else:
        print("  Attributive Clauses: None")
    
    if entry['noun_clauses']:
        print("  Noun Clauses:")
        for clause in entry['noun_clauses']:
            print(f"    - {clause}")
    else:
        print("  Noun Clauses: None")
    
    if entry['adverbial_clauses']:
        print("  Adverbial Clauses:")
        for clause in entry['adverbial_clauses']:
            print(f"    - {clause}")
    else:
        print("  Adverbial Clauses: None")
    
    print("\n" + "-" * 50 + "\n")  # Separator between sentences

with open("/Volumes/Augustus T7/DISSERTATION/clause_annotation_output_added.json", "w") as f:
    json.dump(effective_sentences, f, indent=4, ensure_ascii=False)