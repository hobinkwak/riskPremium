# Risk Premia Estimation
- Fama Macbeth Two Step Regression (1973)
- Three pass method (2021)

본 연구는 팩터의 리스크 프리미엄을 추정하는 방법론에 대해 실증분석을 수행하였다. 기존의 Fama-Macbeth Two-Step Regression과 2021년에 제안된 Three-Step Regression 방법론을 비교하였으며, FF-5 factor와 거시경제 변수(이하, 매크로 팩터)에 대한 리스크 프리미엄을 추정하였다. 매크로 변수의 리스크 프리미엄을 추정하는데 Three-Step Regression이 보다 합리적인 추정량을 산출하였다.


## Implementation
```shell
python main.py
```
```python
from riskpremia import *

Est = Estimator(포트폴리오, 팩터)
# Two-pass
Est.two_pass(adjust_autocorr=True)
# Three-pass
Est.three_pass(max_k=300)
```

## Reference
- Risk, Return, and Equilibrium: Empirical Tests (EF Fama, 1973)
- Asset Pricing with Omitted Factors (S Giglio, 2021)