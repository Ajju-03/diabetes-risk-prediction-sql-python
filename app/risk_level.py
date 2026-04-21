def get_risk_level(data, probability):
    risk_score = 0
    factors = []

    if probability >= 0.7:
        risk_score += 3
    elif probability >= 0.5:
        risk_score += 2

    # Glucose Level (High priority factor)
    if data['Glucose'] >= 140:
        risk_score += 3
        factors.append("High Glucose (>140)")
    elif data['Glucose'] >= 100:
        risk_score += 1
        factors.append("Borderline Glucose")

    # BMI (Body Mass Index)
    if data['BMI'] >= 30:
        risk_score += 2
        factors.append("Obese BMI (>30)")
    elif data['BMI'] >= 25:
        risk_score += 1
        factors.append("Overweight BMI")\
        
    # Age
    if data['Age'] > 45:
        risk_score += 1
    if data['DiabetesPedigreeFunction'] > 0.5:
        risk_score += 1

    if risk_score >= 6 or probability >= 0.7:
        level = "HIGH RISK"
    elif risk_score >= 3 or probability >= 0.4:
        level = "MEDIUM RISK"
    else:
        level = "LOW RISK"
    
    if not factors and probability > 0.3:
        factors.append("Implicit Risk detected by AI analysis")
    elif not factors:
        factors.append("No significant clinical markers identified")

    return {
        "risk_level": level,
        "score": risk_score,
        "identified_factors": factors
    }