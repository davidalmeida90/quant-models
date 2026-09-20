# Second and Third Order Greeks

Vanna, volga, charm, veta, speed, zomma, color and ultima, validated against finite differences and run on a live option chain.

## Run

```bash
pip install -r ../requirements.txt
py -3 model.py
```

Charts are written to `charts/`.

## What is inside

- Files: model.py
- Data: yfinance for the live chain, or --validate to run offline

## Write-up

Full explanation, step by step, with the charts: [https://davidariasfinance.com/scripts/advanced-greeks/](https://davidariasfinance.com/scripts/advanced-greeks/)

## References

- [Haug, The Complete Guide to Option Pricing Formulas](https://www.mheducation.com/highered/product/complete-guide-option-pricing-formulas-haug.html)

## License

MIT, see [LICENSE](../LICENSE).
