# Simple expert system using Python functions and logic

def is_fever(temp):
    return temp > 99.5

def is_cough(symptoms):
    return 'cough' in symptoms

def is_sore_throat(symptoms):
    return 'sore throat' in symptoms

def diagnose(symptoms, temperature):
    rules = {
        "Flu": is_fever(temperature) and is_cough(symptoms),
        "Common Cold": is_cough(symptoms) and is_sore_throat(symptoms),
        "Healthy": not is_fever(temperature) and not is_cough(symptoms)
    }
    for condition, result in rules.items():
        if result:
            return condition
    return "Unknown Condition"

# Example usage:
symptoms = ['cough', 'sore throat']
temperature = 100.2
diagnosis = diagnose(symptoms, temperature)
print("Diagnosis:", diagnosis)
# Example output: Diagnosis: Flu