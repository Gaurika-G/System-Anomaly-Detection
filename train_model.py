from src.train import run_training
from src.evaluate import run_evaluation

if __name__ == "__main__":
    model, X_train, y_train, X_test, y_test = run_training()
    run_evaluation(model, X_train, y_train, X_test, y_test)