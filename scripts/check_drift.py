import pandas as pd
import numpy as np

def calculate_psi(base_series, current_series, feature_type='numeric', buckets=10):
    if feature_type == 'numeric':
        return calculate_numeric_psi(base_series, current_series, buckets)
    else:
        return calculate_categorical_psi(base_series, current_series)

def calculate_numeric_psi(base_series, current_series, buckets=10):
    """PSI для числовых фичей"""
    base = base_series.dropna()
    current = current_series.dropna()
    
    if len(base) == 0 or len(current) == 0:
        return 0.0

    percentiles = np.percentile(base, [100/buckets * i for i in range(buckets + 1)])
    percentiles = np.unique(percentiles)
    
    if len(percentiles) < 2:
        return 0.0
    
    base_hist, _ = np.histogram(base, bins=percentiles)
    current_hist, _ = np.histogram(current, bins=percentiles)
    base_perc = base_hist / len(base) + 0.0001
    current_perc = current_hist / len(current) + 0.0001

    psi = np.sum((current_perc - base_perc) * np.log(current_perc / base_perc))
    
    return float(psi)

def calculate_categorical_psi(base_series, current_series):
    """PSI для категориальных фичей"""
    base = base_series.dropna()
    current = current_series.dropna()
    
    if len(base) == 0 or len(current) == 0:
        return 0.0

    all_categories = set(base.unique()) | set(current.unique())

    base_counts = {cat: 0 for cat in all_categories}
    current_counts = {cat: 0 for cat in all_categories}
    
    for cat in base:
        base_counts[cat] += 1
    for cat in current:
        current_counts[cat] += 1

    base_total = len(base)
    current_total = len(current)
    
    psi_total = 0.0
    for cat in all_categories:
        base_perc = base_counts[cat] / base_total + 0.0001
        current_perc = current_counts[cat] / current_total + 0.0001
        
        psi_total += (current_perc - base_perc) * np.log(current_perc / base_perc)
    
    return float(psi_total)

def check_category_distribution(base_series, current_series):
    """Проверяем изменение распределения категорий"""
    base = base_series.value_counts(normalize=True).sort_index()
    current = current_series.value_counts(normalize=True).sort_index()

    all_cats = base.index.union(current.index)
    base = base.reindex(all_cats, fill_value=0.0001)
    current = current.reindex(all_cats, fill_value=0.0001)

    max_change = abs(current - base).max() * 100
    
    return float(max_change)

def check_drift():
    """Основная функция проверки дрифта"""

    import os
    script_dir = os.path.dirname(__file__)
    data_dir = os.path.join(script_dir, '../data')

    baseline = pd.read_csv(os.path.join(data_dir, 'baseline.csv'))
    current = pd.read_csv(os.path.join(data_dir, 'current.csv'))

    numeric_features = ['age', 'fare', 'family_size']
    categorical_features = ['sex', 'embarked', 'pclass', 'is_alone']
    
    drift_results = {}
    drift_detected = False

    for feature in numeric_features:
        if feature in baseline.columns:
            psi = calculate_psi(baseline[feature], current[feature], 'numeric')
            
            baseline_mean = baseline[feature].mean()
            current_mean = current[feature].mean()
            mean_change = abs((current_mean - baseline_mean) / baseline_mean) if baseline_mean != 0 else 0
            
            drift_results[feature] = {
                'type': 'numeric',
                'psi': float(round(psi, 4)),
                'baseline_mean': float(round(baseline_mean, 2)),
                'current_mean': float(round(current_mean, 2)),
                'mean_change_percent': float(round(mean_change * 100, 1)),
                'drift_detected': psi > 0.2 or mean_change > 0.1
            }
            
            if drift_results[feature]['drift_detected']:
                drift_detected = True
                print(f"{feature}: PSI={psi:.3f}, изменение среднего={mean_change*100:.1f}%")
            else:
                print(f"{feature}: OK")

    for feature in categorical_features:
        if feature in baseline.columns:
            psi = calculate_psi(baseline[feature], current[feature], 'categorical')
            max_change = check_category_distribution(baseline[feature], current[feature])

            base_dist = baseline[feature].value_counts(normalize=True).to_dict()
            current_dist = current[feature].value_counts(normalize=True).to_dict()
            
            drift_results[feature] = {
                'type': 'categorical',
                'psi': float(round(psi, 4)),
                'max_distribution_change_percent': float(round(max_change, 1)),
                'baseline_distribution': base_dist,
                'current_distribution': current_dist,
                'drift_detected': psi > 0.2 or max_change > 10
            }
            
            if drift_results[feature]['drift_detected']:
                drift_detected = True
                print(f"{feature}: PSI={psi:.3f}, макс. изменение={max_change:.1f}%")
            else:
                print(f" {feature}: OK")
    
    return drift_detected

def main():
    if check_drift():
        return True
    else:
        return False

if __name__ == "__main__":
    main()