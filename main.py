import pandas as pd
import numpy as np
import random
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor, Pool
import warnings
import os
warnings.filterwarnings('ignore')

SEED = 322

def set_all_seeds(seed):
    """Установка сидов для всех библиотек"""
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Установлен глобальный seed: {seed}")


def load_data(train_path=r'data/train.csv',
              test_path=r'data/test.csv'):
    """Загрузка данных"""

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    if 'row_id' not in test.columns:
        test['row_id'] = np.arange(len(test))

    return train, test


def prepare_features_improved(df, is_train=True, label_encoders=None, scalers=None):
    """Улучшенная подготовка признаков"""
    df = df.copy()

    if label_encoders is None:
        label_encoders = {}
    if scalers is None:
        scalers = {}

    if 'dt' in df.columns:
        df['dt'] = pd.to_datetime(df['dt'])
        df['year'] = df['dt'].dt.year
        df['month'] = df['dt'].dt.month
        df['dow'] = df['dt'].dt.dayofweek
        df['day_of_month'] = df['dt'].dt.day
        df['day_of_year'] = df['dt'].dt.dayofyear
        df['week_of_year'] = df['dt'].dt.isocalendar().week.astype(int)
        df['quarter'] = df['dt'].dt.quarter

        df['is_weekend'] = (df['dow'] >= 5).astype(int)
        df['is_month_end'] = df['dt'].dt.is_month_end.astype(int)
        df['is_month_start'] = df['dt'].dt.is_month_start.astype(int)

        df['season'] = df['month'] % 12 // 3 + 1
        df['is_summer'] = ((df['month'] >= 6) & (df['month'] <= 8)).astype(int)
        df['is_winter'] = ((df['month'] <= 2) | (df['month'] == 12)).astype(int)

        df['dow_sin'] = np.sin(2 * np.pi * df['dow'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['dow'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    if all(col in df.columns for col in ['avg_temperature', 'avg_humidity']):
        df['temp_humidity'] = df['avg_temperature'] * df['avg_humidity']
        df['feels_like'] = 0.5 * (df['avg_temperature'] + 61.0 + (df['avg_temperature'] - 68.0) * 1.2 + df['avg_humidity'] * 0.094)

    if 'n_stores' in df.columns:
        df['log_stores'] = np.log1p(df['n_stores'])

    if is_train and 'product_id' in df.columns and 'price_p05' in df.columns and 'price_p95' in df.columns:
        product_stats = df.groupby('product_id').agg({
            'price_p05': ['mean', 'median', 'std', 'min', 'max'],
            'price_p95': ['mean', 'median', 'std', 'min', 'max'],
        }).round(2)

        new_columns = []
        for col in product_stats.columns:
            if col[0] == 'price_p05':
                new_columns.append(f'product_p05_{col[1]}')
            elif col[0] == 'price_p95':
                new_columns.append(f'product_p95_{col[1]}')

        product_stats.columns = new_columns

        product_stats['product_price_range'] = product_stats['product_p95_max'] - product_stats['product_p05_min']
        product_stats['product_price_ratio'] = product_stats['product_p95_mean'] / (product_stats['product_p05_mean'] + 1e-8)
        product_stats['product_price_cv'] = product_stats['product_p95_std'] / (product_stats['product_p95_mean'] + 1e-8)
        
        label_encoders['product_stats'] = product_stats

    if not is_train and 'product_stats' in label_encoders:
        product_stats = label_encoders['product_stats']
        for col in product_stats.columns:
            df[f'{col}'] = df['product_id'].map(
                product_stats[col].to_dict()
            ).fillna(0)

    if is_train and 'price_p05' in df.columns and 'price_p95' in df.columns:
        df = df.sort_values(['product_id', 'dt']).reset_index(drop=True)

        for lag in [1, 7, 14]:
            df[f'price_p05_lag_{lag}'] = df.groupby('product_id')['price_p05'].shift(lag)
            df[f'price_p95_lag_{lag}'] = df.groupby('product_id')['price_p95'].shift(lag)

        for window in [7, 14]:
            df[f'price_p05_roll_mean_{window}'] = df.groupby('product_id')['price_p05'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            df[f'price_p95_roll_mean_{window}'] = df.groupby('product_id')['price_p95'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )

    cat_cols = ['management_group_id', 'first_category_id', 
                'second_category_id', 'third_category_id',
                'season', 'quarter']

    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)

    return df, label_encoders, scalers


def train_catboost_model(X_train, y_train, X_val, y_val, cat_indices, alpha, model_name, seed_offset=0):
    """Обучает CatBoost модель"""
    print(f"Training {model_name} with alpha={alpha}...")
    params = {
        'iterations': 1500,
        'learning_rate': 0.05,
        'depth': 7,
        'l2_leaf_reg': 3.0,
        'border_count': 128,
        'loss_function': f'Quantile:alpha={alpha}',
        'eval_metric': f'Quantile:alpha={alpha}',
        'random_seed': SEED + seed_offset,
        'verbose': 100,
        'early_stopping_rounds': 50,
        'use_best_model': True,
        'task_type': 'CPU',
        'thread_count': -1,
        'cat_features': cat_indices,
        'bootstrap_type': 'Bernoulli',
        'subsample': 0.8,
        'random_strength': 1.0,
        'leaf_estimation_method': 'Gradient',
    }

    model = CatBoostRegressor(**params)

    train_pool = Pool(X_train, y_train, cat_features=cat_indices)
    val_pool = Pool(X_val, y_val, cat_features=cat_indices)

    model.fit(
        train_pool,
        eval_set=val_pool,
        plot=False
    )

    return model


def calculate_iou(true_p05, true_p95, pred_p05, pred_p95, epsilon=1e-8):
    """Вычисление IoU"""
    true_lower = true_p05 - epsilon
    true_upper = true_p95 + epsilon
    pred_lower = pred_p05 - epsilon
    pred_upper = pred_p95 + epsilon

    intersection = np.maximum(
        0,
        np.minimum(true_upper, pred_upper) -
        np.maximum(true_lower, pred_lower)
    )

    union = (true_upper - true_lower) + (pred_upper - pred_lower) - intersection
    union = np.maximum(union, epsilon)
    
    iou = intersection / union
    return np.mean(iou)


def create_submission(pred_p05, pred_p95, test_df, submission_path='results/submission.csv'):

    import os

    os.makedirs(os.path.dirname(submission_path) if os.path.dirname(submission_path) else '.', exist_ok=True)

    if 'row_id' not in test_df.columns:
        test_df['row_id'] = np.arange(len(test_df))

    pred_p05 = np.maximum(pred_p05, 0.01)
    pred_p95 = np.maximum(pred_p95, pred_p05 + 0.01)

    pred_p05 = np.round(pred_p05, 2)
    pred_p95 = np.round(pred_p95, 2)
    
    # Создаем сабмишн
    submission_df = pd.DataFrame({
        'row_id': test_df['row_id'].values,
        'price_p05': pred_p05,
        'price_p95': pred_p95
    })

    submission_df = submission_df.sort_values('row_id')
    submission_df.to_csv(submission_path, index=False)


    width_stats = submission_df['price_p95'] - submission_df['price_p05']
    print(f"Interval width - Min: {width_stats.min():.2f}, "
          f"Mean: {width_stats.mean():.2f}, "
          f"Max: {width_stats.max():.2f}")

    return submission_df, submission_path


def main():
    """Главная функция программы"""
    try:
        set_all_seeds(SEED)
        train, test = load_data()


        train_features, label_encoders, scalers = prepare_features_improved(train, is_train=True)
        test_features, _, _ = prepare_features_improved(test, is_train=False, label_encoders=label_encoders)


        feature_cols = [

            'management_group_id', 'first_category_id', 'second_category_id', 'third_category_id',
            'season', 'quarter',

            'month', 'dow', 'day_of_month', 'week_of_year',
            'is_weekend', 'is_month_end', 'is_month_start',
            'dow_sin', 'dow_cos', 'month_sin', 'month_cos',

            'avg_temperature', 'avg_humidity', 'avg_wind_level', 'precpt',
            'temp_humidity', 'feels_like',

            'n_stores', 'log_stores',

            'holiday_flag', 'activity_flag',
        ]

        product_stat_cols = [col for col in train_features.columns if col.startswith('product_')]
        feature_cols.extend(product_stat_cols)

        for col in train_features.columns:
            if any(x in col for x in ['lag_', 'roll_']) and col not in feature_cols:
                feature_cols.append(col)

        feature_cols = [f for f in feature_cols
                       if f in train_features.columns and f in test_features.columns]
        
        print(f"Всего признаков: {len(feature_cols)}")


        train_features = train_features.sort_values(['product_id', 'dt']).reset_index(drop=True)
        val_size = int(len(train_features) * 0.2)
        train_data = train_features.iloc[:-val_size]
        val_data = train_features.iloc[-val_size:]

        X_train_raw = train_data[feature_cols].copy()
        X_val_raw = val_data[feature_cols].copy()
        X_test_raw = test_features[feature_cols].copy()
        
        y_train_p05 = train_data['price_p05'].values
        y_train_p95 = train_data['price_p95'].values
        y_val_p05 = val_data['price_p05'].values
        y_val_p95 = val_data['price_p95'].values

        cat_cols_for_cb = ['management_group_id', 'first_category_id', 
                          'second_category_id', 'third_category_id',
                          'season', 'quarter']
        cat_cols_for_cb = [col for col in cat_cols_for_cb if col in feature_cols]
        
        num_cols = [col for col in feature_cols if col not in cat_cols_for_cb]
        
        print(f"Категориальные признаки: {len(cat_cols_for_cb)}")
        print(f"Числовые признаки: {len(num_cols)}")

        imputer = SimpleImputer(strategy='median')

        if num_cols:
            X_train_num = pd.DataFrame(
                imputer.fit_transform(X_train_raw[num_cols]),
                columns=num_cols,
                index=X_train_raw.index
            )
            X_val_num = pd.DataFrame(
                imputer.transform(X_val_raw[num_cols]),
                columns=num_cols,
                index=X_val_raw.index
            )
            X_test_num = pd.DataFrame(
                imputer.transform(X_test_raw[num_cols]),
                columns=num_cols,
                index=X_test_raw.index
            )

            X_train_cat = X_train_raw[cat_cols_for_cb].copy()
            X_val_cat = X_val_raw[cat_cols_for_cb].copy()
            X_test_cat = X_test_raw[cat_cols_for_cb].copy()

            X_train = pd.concat([X_train_num, X_train_cat], axis=1)[feature_cols]
            X_val = pd.concat([X_val_num, X_val_cat], axis=1)[feature_cols]
            X_test = pd.concat([X_test_num, X_test_cat], axis=1)[feature_cols]
        else:
            X_train = X_train_raw.copy()
            X_val = X_val_raw.copy()
            X_test = X_test_raw.copy()

        for col in cat_cols_for_cb:
            X_train[col] = X_train[col].astype(str)
            X_val[col] = X_val[col].astype(str)
            X_test[col] = X_test[col].astype(str)

            X_train[col] = X_train[col].replace('nan', 'missing').replace('None', 'missing')
            X_val[col] = X_val[col].replace('nan', 'missing').replace('None', 'missing')
            X_test[col] = X_test[col].replace('nan', 'missing').replace('None', 'missing')

        cat_indices = [i for i, col in enumerate(feature_cols) if col in cat_cols_for_cb]

        print(f"Train shape: {X_train.shape}")
        print(f"Validation shape: {X_val.shape}")
        print(f"Test shape: {X_test.shape}")


        models_p05 = []
        for i in range(2):
            model = train_catboost_model(X_train, y_train_p05, X_val, y_val_p05, cat_indices, 0.05,
                                        f"p05 модель {i+1}", seed_offset=i*10)
            models_p05.append(model)

        models_p95 = []
        for i in range(2):
            model = train_catboost_model(X_train, y_train_p95, X_val, y_val_p95, cat_indices, 0.95,
                                        f"p95 модель {i+1}", seed_offset=i*10 + 5)
            models_p95.append(model)




        val_preds_p05 = [model.predict(X_val) for model in models_p05]
        val_preds_p95 = [model.predict(X_val) for model in models_p95]

        val_weights_p05 = [1.0, 0.8]
        val_weights_p95 = [1.0, 0.8]
        
        val_pred_p05_weighted = np.average(val_preds_p05, axis=0, weights=val_weights_p05)
        val_pred_p95_weighted = np.average(val_preds_p95, axis=0, weights=val_weights_p95)

        val_pred_p05 = np.maximum(val_pred_p05_weighted, 0.01)
        val_pred_p95 = np.maximum(val_pred_p95_weighted, val_pred_p05 + 0.01)
        
        initial_iou = calculate_iou(y_val_p05, y_val_p95, val_pred_p05, val_pred_p95)
        print(f"Начальное IoU ансамбля: {initial_iou:.6f}")

        print("\nВыполнение оптимизации...")
        
        best_iou = initial_iou
        best_params = {
            'width_factor': 1.0,
            'shift_factor': 0.0,
            'asymmetry': 0.0,
            'method': 'midpoint'
        }
        best_pred_p05 = val_pred_p05.copy()
        best_pred_p95 = val_pred_p95.copy()

        methods = ['midpoint', 'lower', 'upper', 'both']
        
        np.random.seed(SEED)
        for method in methods:
            for width_factor in np.linspace(0.7, 1.5, 9):
                for shift_factor in np.linspace(-0.1, 0.1, 5):
                    for asymmetry in np.linspace(-0.1, 0.1, 5):
                        
                        if method == 'midpoint':

                            val_mid = (val_pred_p05 + val_pred_p95) / 2
                            val_width = val_pred_p95 - val_pred_p05
                            
                            adjusted_width = val_width * width_factor
                            adjusted_mid = val_mid * (1 + shift_factor)
                            
                            adjusted_p05 = adjusted_mid - adjusted_width / 2
                            adjusted_p95 = adjusted_mid + adjusted_width / 2
                        
                        elif method == 'lower':

                            val_mid = (val_pred_p05 + val_pred_p95) / 2
                            val_width = val_pred_p95 - val_pred_p05
                            
                            adjusted_width = val_width * width_factor
                            adjusted_p05 = val_pred_p05 * (1 + shift_factor - asymmetry)
                            adjusted_p95 = adjusted_p05 + adjusted_width
                        
                        elif method == 'upper':

                            val_mid = (val_pred_p05 + val_pred_p95) / 2
                            val_width = val_pred_p95 - val_pred_p05

                            adjusted_width = val_width * width_factor
                            adjusted_p95 = val_pred_p95 * (1 + shift_factor + asymmetry)
                            adjusted_p05 = adjusted_p95 - adjusted_width

                        else:
                            adjusted_p05 = val_pred_p05 * (1 + shift_factor - asymmetry)
                            adjusted_p95 = val_pred_p95 * (1 + shift_factor + asymmetry)
                            adjusted_width = adjusted_p95 - adjusted_p05

                            if width_factor != 1.0:
                                adjusted_mid = (adjusted_p05 + adjusted_p95) / 2
                                adjusted_p05 = adjusted_mid - (adjusted_width * width_factor) / 2
                                adjusted_p95 = adjusted_mid + (adjusted_width * width_factor) / 2

                        adjusted_p05 = np.maximum(adjusted_p05, 0.01)
                        adjusted_p95 = np.maximum(adjusted_p95, adjusted_p05 + 0.01)

                        iou = calculate_iou(y_val_p05, y_val_p95, adjusted_p05, adjusted_p95)

                        if iou > best_iou:
                            best_iou = iou
                            best_params = {
                                'width_factor': width_factor,
                                'shift_factor': shift_factor,
                                'asymmetry': asymmetry,
                                'method': method
                            }
                            best_pred_p05 = adjusted_p05.copy()
                            best_pred_p95 = adjusted_p95.copy()

        print(f"Лучший метод: {best_params['method']}")
        print(f"Лучшие параметры: width={best_params['width_factor']:.3f}, "
              f"shift={best_params['shift_factor']:.3f}, asym={best_params['asymmetry']:.3f}")
        print(f"Лучшее IoU: {best_iou:.6f}")
        print(f"Улучшение IoU: {best_iou - initial_iou:.6f}")

        test_preds_p05 = [model.predict(X_test) for model in models_p05]
        test_preds_p95 = [model.predict(X_test) for model in models_p95]

        test_pred_p05 = np.average(test_preds_p05, axis=0, weights=val_weights_p05)
        test_pred_p95 = np.average(test_preds_p95, axis=0, weights=val_weights_p95)

        print(f"Применение лучшего метода калибровки: {best_params['method']}")

        if best_params['method'] == 'midpoint':
            test_mid = (test_pred_p05 + test_pred_p95) / 2
            test_width = test_pred_p95 - test_pred_p05

            adjusted_width = test_width * best_params['width_factor']
            adjusted_mid = test_mid * (1 + best_params['shift_factor'])

            test_pred_p05 = adjusted_mid - adjusted_width / 2
            test_pred_p95 = adjusted_mid + adjusted_width / 2

        elif best_params['method'] == 'lower':
            test_mid = (test_pred_p05 + test_pred_p95) / 2
            test_width = test_pred_p95 - test_pred_p05

            adjusted_width = test_width * best_params['width_factor']
            test_pred_p05 = test_pred_p05 * (1 + best_params['shift_factor'] - best_params['asymmetry'])
            test_pred_p95 = test_pred_p05 + adjusted_width

        elif best_params['method'] == 'upper':
            test_mid = (test_pred_p05 + test_pred_p95) / 2
            test_width = test_pred_p95 - test_pred_p05

            adjusted_width = test_width * best_params['width_factor']
            test_pred_p95 = test_pred_p95 * (1 + best_params['shift_factor'] + best_params['asymmetry'])
            test_pred_p05 = test_pred_p95 - adjusted_width

        else:
            test_pred_p05 = test_pred_p05 * (1 + best_params['shift_factor'] - best_params['asymmetry'])
            test_pred_p95 = test_pred_p95 * (1 + best_params['shift_factor'] + best_params['asymmetry'])

            test_width = test_pred_p95 - test_pred_p05
            if best_params['width_factor'] != 1.0:
                test_mid = (test_pred_p05 + test_pred_p95) / 2
                test_pred_p05 = test_mid - (test_width * best_params['width_factor']) / 2
                test_pred_p95 = test_mid + (test_width * best_params['width_factor']) / 2

        print("Применение интеллектуальной постобработки...")

        train_p05_mean = train['price_p05'].mean()
        train_p95_mean = train['price_p95'].mean()
        train_width_mean = (train['price_p95'] - train['price_p05']).mean()

        test_p05_mean = test_pred_p05.mean()
        test_p95_mean = test_pred_p95.mean()

        test_pred_p05 = test_pred_p05 * (train_p05_mean / test_p05_mean) * 0.3 + test_pred_p05 * 0.7
        test_pred_p95 = test_pred_p95 * (train_p95_mean / test_p95_mean) * 0.3 + test_pred_p95 * 0.7

        train_widths = train['price_p95'] - train['price_p05']
        min_width = np.percentile(train_widths, 10)
        max_width = np.percentile(train_widths, 90)

        test_widths = test_pred_p95 - test_pred_p05

        width_mask_small = test_widths < min_width
        if width_mask_small.any():
            mid = (test_pred_p05[width_mask_small] + test_pred_p95[width_mask_small]) / 2
            test_pred_p05[width_mask_small] = mid - min_width / 2
            test_pred_p95[width_mask_small] = mid + min_width / 2

        width_mask_large = test_widths > max_width
        if width_mask_large.any():
            mid = (test_pred_p05[width_mask_large] + test_pred_p95[width_mask_large]) / 2
            test_pred_p05[width_mask_large] = mid - max_width / 2
            test_pred_p95[width_mask_large] = mid + max_width / 2

        def winsorize(values, lower=0.01, upper=0.99):
            lower_bound = np.percentile(values, lower * 100)
            upper_bound = np.percentile(values, upper * 100)
            return np.clip(values, lower_bound, upper_bound)

        test_pred_p05 = winsorize(test_pred_p05, 0.01, 0.99)
        test_pred_p95 = winsorize(test_pred_p95, 0.01, 0.99)
        submission_df, submission_path = create_submission(
            pred_p05=test_pred_p05,
            pred_p95=test_pred_p95,
            test_df=test,
            submission_path='results/submission.csv'
        )

    except Exception as e:
        print(f"Ошибка в выполнении: {e}")
        import traceback
        traceback.print_exc()
        raise

    return submission_df


if __name__ == "__main__":
    submission = main()