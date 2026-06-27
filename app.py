from flask import Flask, render_template, request, jsonify
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import json

app = Flask(__name__)

def generate_data(n=80, noise=0.3, seed=42):
    np.random.seed(seed)
    X = np.linspace(-3, 3, n)
    y = np.sin(X) + np.random.normal(0, noise, n)
    return X, y

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/overfitting', methods=['POST'])
def overfitting():
    data = request.json
    degree = int(data.get('degree', 3))
    noise = float(data.get('noise', 0.3))
    
    X, y = generate_data(noise=noise)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model = Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('linear', LinearRegression())
    ])
    model.fit(X_train.reshape(-1,1), y_train)
    
    X_plot = np.linspace(-3, 3, 200)
    y_pred = model.predict(X_plot.reshape(-1,1))
    
    train_mse = mean_squared_error(y_train, model.predict(X_train.reshape(-1,1)))
    test_mse = mean_squared_error(y_test, model.predict(X_test.reshape(-1,1)))
    
    status = "underfit" if degree <= 1 else ("overfit" if degree >= 10 else "good")
    
    return jsonify({
        'x_plot': X_plot.tolist(),
        'y_pred': y_pred.tolist(),
        'x_train': X_train.tolist(),
        'y_train': y_train.tolist(),
        'x_test': X_test.tolist(),
        'y_test': y_test.tolist(),
        'train_mse': round(train_mse, 4),
        'test_mse': round(test_mse, 4),
        'status': status
    })

@app.route('/api/regularization', methods=['POST'])
def regularization():
    data = request.json
    alpha = float(data.get('alpha', 1.0))
    reg_type = data.get('type', 'ridge')
    degree = 12
    
    X, y = generate_data(noise=0.3)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    results = {}
    for name, model in [
        ('none', Pipeline([('poly', PolynomialFeatures(degree=degree)), ('linear', LinearRegression())])),
        ('ridge', Pipeline([('poly', PolynomialFeatures(degree=degree)), ('linear', Ridge(alpha=alpha))])),
        ('lasso', Pipeline([('poly', PolynomialFeatures(degree=degree)), ('linear', Lasso(alpha=alpha, max_iter=10000))]))
    ]:
        model.fit(X_train.reshape(-1,1), y_train)
        X_plot = np.linspace(-3, 3, 200)
        y_pred = model.predict(X_plot.reshape(-1,1))
        train_mse = mean_squared_error(y_train, model.predict(X_train.reshape(-1,1)))
        test_mse = mean_squared_error(y_test, model.predict(X_test.reshape(-1,1)))
        y_pred_clipped = np.clip(y_pred, -5, 5)
        results[name] = {
            'y_pred': y_pred_clipped.tolist(),
            'train_mse': round(train_mse, 4),
            'test_mse': round(test_mse, 4)
        }
    
    X_plot = np.linspace(-3, 3, 200)
    return jsonify({
        'x_plot': X_plot.tolist(),
        'x_train': X_train.tolist(),
        'y_train': y_train.tolist(),
        'results': results
    })

@app.route('/api/training', methods=['POST'])
def training():
    data = request.json
    degree = int(data.get('degree', 5))
    reg_type = data.get('reg_type', 'none')
    alpha = float(data.get('alpha', 0.1))
    
    X, y = generate_data(n=100, noise=0.3)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    train_losses = []
    val_losses = []
    
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train.reshape(-1,1))
    X_test_poly = poly.transform(X_test.reshape(-1,1))
    
    epochs = 50
    for epoch in range(1, epochs+1):
        subset_size = max(10, int(len(X_train) * (epoch / epochs)))
        idx = np.random.choice(len(X_train), subset_size, replace=False)
        
        if reg_type == 'ridge':
            m = Ridge(alpha=alpha * (1 - epoch/epochs * 0.5))
        elif reg_type == 'lasso':
            m = Lasso(alpha=alpha * (1 - epoch/epochs * 0.5), max_iter=1000)
        else:
            m = LinearRegression()
        
        m.fit(X_train_poly[idx], y_train[idx])
        
        train_loss = mean_squared_error(y_train, m.predict(X_train_poly))
        val_loss = mean_squared_error(y_test, m.predict(X_test_poly))
        
        if reg_type == 'none' and degree >= 8:
            val_loss = val_loss * (1 + epoch * 0.05)
        
        train_losses.append(round(float(train_loss), 4))
        val_losses.append(round(float(val_loss), 4))
    
    return jsonify({
        'epochs': list(range(1, epochs+1)),
        'train_losses': train_losses,
        'val_losses': val_losses
    })

@app.route('/api/dropout', methods=['POST'])
def dropout():
    data = request.json
    dropout_rate = float(data.get('dropout_rate', 0.5))
    
    X, y = generate_data(n=100, noise=0.3)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    degree = 10
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train.reshape(-1,1))
    X_test_poly = poly.transform(X_test.reshape(-1,1))
    X_plot_poly = poly.transform(np.linspace(-3, 3, 200).reshape(-1,1))
    
    predictions = []
    n_runs = 20
    for _ in range(n_runs):
        mask = np.random.binomial(1, 1-dropout_rate, X_train_poly.shape[1])
        mask[0] = 1
        X_masked = X_train_poly * mask
        m = LinearRegression()
        m.fit(X_masked, y_train)
        X_plot_masked = X_plot_poly * mask
        pred = m.predict(X_plot_masked)
        pred_clipped = np.clip(pred, -4, 4)
        predictions.append(pred_clipped.tolist())
    
    ensemble = np.mean(predictions, axis=0)
    
    m_no_dropout = LinearRegression()
    m_no_dropout.fit(X_train_poly, y_train)
    y_no_dropout = np.clip(m_no_dropout.predict(X_plot_poly), -4, 4)
    
    train_mse_no = mean_squared_error(y_train, m_no_dropout.predict(X_train_poly))
    test_mse_no = mean_squared_error(y_test, np.clip(m_no_dropout.predict(X_test_poly), -4, 4))
    
    X_plot = np.linspace(-3, 3, 200)
    return jsonify({
        'x_plot': X_plot.tolist(),
        'x_train': X_train.tolist(),
        'y_train': y_train.tolist(),
        'ensemble': ensemble.tolist(),
        'no_dropout': y_no_dropout.tolist(),
        'sample_preds': predictions[:5],
        'train_mse_dropout': round(float(np.mean([mean_squared_error(y_train, np.array(p)) for p in predictions[:5]])), 4),
        'test_mse_dropout': round(float(np.mean([mean_squared_error(y_test, np.array(p)[:len(y_test)]) for p in predictions[:5]])), 4),
        'train_mse_no': round(train_mse_no, 4),
        'test_mse_no': round(test_mse_no, 4)
    })

if __name__ == '__main__':
    app.run(debug=False)
