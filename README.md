# Risk Premia Estimation
- Fama Macbeth Two Step Regression (1973)
- Three pass method (2021)

This study empirically analyzes methodologies for estimating risk premiums of factors. It compares the traditional Fama-Macbeth Two-Step Regression with the Three-Step Regression proposed in 2021, estimating risk premiums for the FF-5 factors and macroeconomic variables (hereinafter, macro factors). The Three-Step Regression produced more reasonable estimates when estimating the risk premiums of macro factors.


## Implementation
```shell
python main.py
```
```python
from riskpremia import *

Est = Estimator(portfolio, factor)
# Two-pass
Est.two_pass(adjust_autocorr=True)
# Three-pass
Est.three_pass(max_k=300)
```

## Reference
- Risk, Return, and Equilibrium: Empirical Tests (EF Fama, 1973)
- Asset Pricing with Omitted Factors (S Giglio, 2021)
