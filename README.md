# End-to-End ML Trading Research Framework

This repository contains a modular, full-stack Python framework for financial data science and trading research. It is designed to handle the entire pipeline, from data ingestion and storage to advanced feature selection and hyperparameter optimization.

### System Architecture & Workflow

The framework is designed as a sequential pipeline that transforms raw market data into an optimized and evaluated machine learning model.

```
[External APIs (e.g., IBKR)]
        |
        v (Fetched by stockml/api)
[PostgreSQL Database (Managed by stockml/sql)] <--- (Foundation for all modules)
        |
        v (Processed by stockml/dataset)
[Engineered Feature Sets (TA-Lib, Pandas-TA, etc.)]
        |
        v (Analyzed by stockml/optimizations)
[Multiple Feature Selection Methods (Lasso, RF, MRMR, SFS)]
        |
        v (Selects optimal features for...)
[ML Model (Scikit-learn / River ML)]
        |
        v (Tuned by stockml/optimizations using Optuna)
[Optimized & Evaluated Model]
```

---

## Skills & Technologies Demonstrated

### Python Data Analysis & Processing
- **Complex Data Manipulation:** Pandas
- **Numerical Computations:** NumPy
- **Visualization:** Plotly
- **Statistical Analysis:** Statsmodels
- **Time Series Handling:** Custom data transformation pipelines for financial data.

### Feature Engineering & Selection
- **Automated Feature Generation:** TA-Lib, Pandas-TA
- **Comprehensive Selection Suite:**
  - L1 Regularization (Lasso)
  - Random Forest Importance
  - Sequential Feature Selection (mlxtend)
  - MRMR (Minimum Redundancy Maximum Relevance)

### Machine Learning Implementation
- **Model Integration:** Scikit-learn pipelines, Online learning with River ML
- **Hyperparameter Optimization:** Optuna with custom, domain-specific objective functions.

### Infrastructure & Software Engineering
- **Database:** PostgreSQL with SQLAlchemy ORM for robust data management.
- **API Integration:** RESTful API client for data ingestion.
- **Best Practices:** Modular code, type hints, secure credential management (`.env`), error handling, and version control.

---

## Project Structure
```python
├── stockml/
│   ├── dataset/        # Data processing and transformation
│   ├── optimizations/  # ML optimization and feature selection
│   ├── api/            # External API integration
│   ├── sql/            # SQL Database management
│   └── utils/          # Helper functions and utilities
```

---
## Getting Started

*(This section is optional but highly recommended for a professional look)*

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/lkbaldridge/End-to-End-Python-Data-Pipeline---A-Complete-ML-Trading-Framework.git
    cd End-to-End-Python-Data-Pipeline---A-Complete-ML-Trading-Framework
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Set up configuration:**
    *   Create a `.env` file in the root directory.
    *   Add your API keys and database credentials based on the `env.template` file.

---

## Contact
Lance Kendrick F. Baldridge
- **LinkedIn:** [linkedin.com/in/lance-baldridge-2a291097](https://www.linkedin.com/in/lance-baldridge-2a291097/)
- **Email:** lance_baldridge@outlook.com
4.  **Action-Oriented:** Adding a "Getting Started" section (even if no one uses it) is a hallmark of a professional, well-documented project.

This merged version is a significant improvement. It tells a complete story, from high-level architecture down to the specific libraries you mastered. **Use this one.**
