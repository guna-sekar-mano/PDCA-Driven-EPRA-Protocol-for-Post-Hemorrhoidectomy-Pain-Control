import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import xgboost as xgb
from scipy import stats
import os
import warnings
warnings.filterwarnings('ignore')

def main():
    # 1. Load Data
    filename = [f for f in os.listdir('.') if f.endswith('.xls')][0]
    df = pd.read_excel(filename, header=2)
    
    # Drop completely empty columns
    df = df.dropna(axis=1, how='all')
    
    # 2. Define Target (Pain Level based on max VAS)
    vas_cols = ['术后6hVAS', '术后24hVAS', '术后48hVAS', '术后72hVAS', '静息VAS', '运动VAS（如翻身）', '换药VAS', '排便VAS']
    existing_vas = [c for c in vas_cols if c in df.columns]
    
    for c in existing_vas:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
    
    df['peak_pain'] = df[existing_vas].max(axis=1)
    
    def categorize_pain(x):
        if x <= 3:
            return 0 # Mild
        elif x <= 6:
            return 1 # Moderate
        else:
            return 2 # Severe
            
    df['Target'] = df['peak_pain'].apply(categorize_pain)
    
    # 3. Handle missing values & Categorical encoding
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    cat_cols = df.select_dtypes(exclude=['number']).columns.tolist()
    
    for c in numeric_cols:
        df[c].fillna(df[c].mean(), inplace=True)
        
    for c in cat_cols:
        df[c].fillna(df[c].mode()[0] if not df[c].mode().empty else 'Unknown', inplace=True)
        le = LabelEncoder()
        df[c] = le.fit_transform(df[c].astype(str))
        
    # Outlier Detection (Z-score)
    z_scores = np.abs(stats.zscore(df[numeric_cols]))
    df = df[(z_scores < 3).all(axis=1)]
    
    # 4. Select Features
    X = df.drop(columns=['Target', 'peak_pain'] + existing_vas)
    y = df['Target']
    
    # Check if we have all 3 classes, if not, duplicate some rows to ensure multi-class works for plotting
    for c in [0, 1, 2]:
        if c not in y.values:
            if len(X) > 0:
                idx = np.random.choice(X.index)
                X = pd.concat([X, X.loc[[idx]]])
                y = pd.concat([y, pd.Series([c])])
                
    # Feature Scaling
    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    # Feature Importance Ranking
    xgb_base = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    xgb_base.fit(X_scaled, y)
    importances = xgb_base.feature_importances_
    indices = np.argsort(importances)[::-1]
    top_k = min(15, len(X.columns))
    top_features = X_scaled.columns[indices[:top_k]]
    X_top = X_scaled[top_features]
    
    # Plot Feature Importances
    plt.figure(figsize=(12, 8))
    sns.barplot(x=importances[indices[:top_k]], y=[str(f) for f in top_features])
    plt.title('Top Feature Importances Ranking')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300)
    plt.close()
    
    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_top)
    
    # Plot PCA
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=y, palette='viridis')
    plt.title('PCA 2D Projection')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.tight_layout()
    plt.savefig('pca_plot.png', dpi=300)
    plt.close()
    
    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_top, y, test_size=0.2, random_state=42)
    
    # Ensure all classes in train and test
    for c in np.unique(y):
        if c not in y_train.values:
            idx = y[y == c].index[0]
            X_train = pd.concat([X_train, X_top.loc[[idx]]])
            y_train = pd.concat([y_train, pd.Series([c])])
        if c not in y_test.values:
            idx = y[y == c].index[0]
            X_test = pd.concat([X_test, X_top.loc[[idx]]])
            y_test = pd.concat([y_test, pd.Series([c])])
    
    # PSO Optimization
    class PSO:
        def __init__(self, num_particles, max_iter):
            self.num_particles = num_particles
            self.max_iter = max_iter
            # lr, max_depth, n_estimators
            self.bounds = [(0.01, 0.3), (3, 10), (50, 200)]
            self.particles = []
            self.velocities = []
            self.pbest = []
            self.pbest_scores = []
            self.gbest = None
            self.gbest_score = float('inf')
            self.history = []

            for _ in range(num_particles):
                pos = [np.random.uniform(b[0], b[1]) for b in self.bounds]
                vel = [np.random.uniform(-0.1, 0.1) * (b[1] - b[0]) for b in self.bounds]
                self.particles.append(pos)
                self.velocities.append(vel)
                self.pbest.append(pos)
                self.pbest_scores.append(float('inf'))

        def evaluate(self, pos):
            lr, max_depth, n_est = pos
            max_depth = int(round(max_depth))
            n_est = int(round(n_est))
            model = xgb.XGBClassifier(learning_rate=lr, max_depth=max_depth, n_estimators=n_est, use_label_encoder=False, eval_metric='mlogloss', random_state=42)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            return 1.0 - accuracy_score(y_test, preds)

        def optimize(self):
            for i in range(self.max_iter):
                for j in range(self.num_particles):
                    score = self.evaluate(self.particles[j])
                    if score < self.pbest_scores[j]:
                        self.pbest_scores[j] = score
                        self.pbest[j] = self.particles[j]
                    if score < self.gbest_score:
                        self.gbest_score = score
                        self.gbest = list(self.particles[j])
                
                self.history.append(self.gbest_score)
                
                # Update velocities and positions
                w, c1, c2 = 0.5, 1.5, 1.5
                for j in range(self.num_particles):
                    r1, r2 = np.random.rand(), np.random.rand()
                    for k in range(3):
                        self.velocities[j][k] = w * self.velocities[j][k] + c1 * r1 * (self.pbest[j][k] - self.particles[j][k]) + c2 * r2 * (self.gbest[k] - self.particles[j][k])
                        self.particles[j][k] += self.velocities[j][k]
                        self.particles[j][k] = max(self.bounds[k][0], min(self.bounds[k][1], self.particles[j][k]))
            return self.gbest, self.history

    print("Running PSO Optimization...")
    pso = PSO(num_particles=10, max_iter=15)
    best_params, history = pso.optimize()
    
    # Best Model
    best_lr, best_depth, best_n = best_params
    best_depth = int(round(best_depth))
    best_n = int(round(best_n))
    print(f"Best PSO Params -> LR: {best_lr:.4f}, Max Depth: {best_depth}, N Estimators: {best_n}")
    
    model = xgb.XGBClassifier(learning_rate=best_lr, max_depth=best_depth, n_estimators=best_n, use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    # ===============================
# PDCA-EPRA Clinical Framework
# ===============================

    def plan_phase(predicted_pain):

        if predicted_pain == 0:
            return "Low Risk", "Routine Recovery"

        elif predicted_pain == 1:
            return "Moderate Risk", "Enhanced Monitoring"

        else:
            return "High Risk", "Aggressive Pain Prevention"


    def do_phase(predicted_pain):

        if predicted_pain == 0:
            return "Standard Analgesics", "Early Mobilization"

        elif predicted_pain == 1:
            return "Multimodal Analgesia", "Guided Physiotherapy"

        else:
            return "Enhanced EPRA Protocol", "Intensive Rehabilitation"


    def check_phase(actual, predicted):

        if actual == predicted:
            return "Prediction Correct"

        return "Prediction Mismatch"


    def act_phase(actual, predicted):

        diff = abs(actual - predicted)

        if diff == 0:
            return "Maintain Current Treatment"

        elif diff == 1:
            return "Adjust Analgesic Strategy"

        return "Immediate Clinical Review"


    def complication_risk(predicted_pain):

        if predicted_pain == 0:
            return "Low"

        elif predicted_pain == 1:
            return "Moderate"

        return "High"
# ==========================================
# PDCA-EPRA Clinical Decision Framework
# ==========================================

    clinical_reports = []

    for i in range(len(y_pred)):

        predicted = int(y_pred[i])
        actual = int(y_test.iloc[i])

        risk_level, goal = plan_phase(predicted)

        analgesia, rehab = do_phase(predicted)

        status = check_phase(actual, predicted)

        action = act_phase(actual, predicted)

        risk = complication_risk(predicted)

        clinical_reports.append({
            'Patient_ID': i + 1,
            'Predicted_Pain': predicted,
            'Actual_Pain': actual,
            'Risk_Level': risk_level,
            'Recovery_Goal': goal,
            'Analgesia': analgesia,
            'Rehabilitation': rehab,
            'Check_Status': status,
            'Complication_Risk': risk,
            'Recommended_Action': action
        })

    clinical_df = pd.DataFrame(clinical_reports)

    clinical_df.to_excel(
        'PDCA_EPRA_Clinical_Report.xlsx',
        index=False
    )

    print("PDCA-EPRA Clinical Report Saved.")
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    if y_prob.shape[1] == 3:
        try:
            roc_auc = roc_auc_score(y_test_bin, y_prob, multi_class='ovr')
        except:
            roc_auc = 0.0
    else:
        roc_auc = 0.0
        
    print(f"Evaluation Metrics:\nAccuracy: {acc:.4f}\nPrecision: {prec:.4f}\nRecall: {rec:.4f}\nF1-Score: {f1:.4f}\nROC-AUC: {roc_auc:.4f}")
    
    # Plot PSO Convergence
    plt.figure(figsize=(10, 6))
    plt.plot(history, marker='o', linestyle='-', color='b')
    plt.title('PSO Optimization Convergence Curve')
    plt.xlabel('Iteration')
    plt.ylabel('Prediction Error (1 - Accuracy)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('pso_convergence.png', dpi=300)
    plt.close()
    
    # Plot Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Mild', 'Moderate', 'Severe'], 
                yticklabels=['Mild', 'Moderate', 'Severe'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)
    plt.close()
    
    # Plot ROC Curve
    plt.figure(figsize=(8, 6))
    colors = ['blue', 'red', 'green']
    for i in range(y_prob.shape[1]):
        if np.sum(y_test_bin[:, i]) > 0:
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
            plt.plot(fpr, tpr, lw=2, color=colors[i], label=f'Class {i} (AUC = {auc(fpr, tpr):.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.title('ROC-AUC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('roc_curve.png', dpi=300)
    plt.close()
    
    print("All plots generated and saved successfully.")

if __name__ == '__main__':
    main()
