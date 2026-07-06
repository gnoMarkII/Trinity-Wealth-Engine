```json
{
  "overall_regime": "Dovish Reflation",
  "regime_probabilities": {
    "Dovish Reflation": 60,
    "Soft Landing / Goldilocks": 25,
    "Stagflationary Pressure": 10,
    "Recession / Deflation": 5
  },
  "regime_evidence": [
    {"dimension": "Growth", "signal": "Expansion", "evidence": "GDP 2.5%", "conflict": "-", "confidence": "high", "observable_refs": ["obs_gdp", "obs_indpro"], "source_refs": ["us_macro.md"]},
    {"dimension": "Inflation", "signal": "Disinflation", "evidence": "CPI 2.4%", "conflict": "-", "confidence": "high", "observable_refs": ["obs_cpi", "obs_pce"], "source_refs": ["us_macro.md"]},
    {"dimension": "Liquidity", "signal": "Easing", "evidence": "Fed Funds 4.75%", "conflict": "-", "confidence": "high", "observable_refs": ["obs_fedfunds", "obs_nfci"], "source_refs": ["us_macro.md"]},
    {"dimension": "Risk sentiment", "signal": "Bullish", "evidence": "VIX 15.2", "conflict": "-", "confidence": "high", "observable_refs": ["obs_vix", "obs_pcall"], "source_refs": ["us_macro.md"]},
    {"dimension": "Regional Inflation Risk", "signal": "Elevated", "evidence": "EU CPI 2.8%", "conflict": "ความผันผวนของราคาพลังงานในยุโรปอาจทำให้เงินเฟ้อลดลงช้ากว่าสหรัฐฯ", "confidence": "medium", "observable_refs": ["obs_eu_cpi"], "source_refs": ["eu_macro.md"]}
  ],
  "asset_allocation": [
    {
      "asset_class": "US Equities",
      "asset_bucket": "equities",
      "stance": "OVERWEIGHT",
      "confidence": "HIGH",
      "allocation_delta": "+3% vs benchmark",
      "benchmark_ref": "S&P 500",
      "time_horizon": "3-6 months",
      "rationale": "ได้รับแรงหนุนจากดอกเบี้ยขาลงและผลประกอบการกลุ่มเทคโนโลยีแข็งแกร่ง",
      "why_not_high": "-",
      "supporting_data": ["EPS growth 10%", "Real Yield 1.8%"],
      "observable_refs": ["obs_001", "obs_002"],
      "source_refs": ["fred_macro.md", "us_sector.md"]
    },
    {
      "asset_class": "US Treasury Bonds",
      "asset_bucket": "fixed_income",
      "stance": "NEUTRAL",
      "confidence": "MEDIUM",
      "allocation_delta": "0% vs benchmark",
      "benchmark_ref": "10Y Treasury",
      "time_horizon": "3-6 months",
      "rationale": "ผลตอบแทนยังน่าสนใจแต่มีความเสี่ยงจากปริมาณการออกพันธบัตรใหม่",
      "why_not_high": "ความผันผวนของอุปทานพันธบัตรสหรัฐฯ ในระยะยาว",
      "supporting_data": ["10Y Yield 4.2%"],
      "observable_refs": ["obs_003", "obs_004"],
      "source_refs": ["fred_macro.md", "global_snapshot.md"]
    },
    {
      "asset_class": "Precious Metals (Gold)",
      "asset_bucket": "commodities",
      "stance": "OVERWEIGHT",
      "confidence": "HIGH",
      "allocation_delta": "+2% vs benchmark",
      "benchmark_ref": "Spot Gold",
      "time_horizon": "3-6 months",
      "rationale": "ได้รับแรงหนุนจากการลดลงของ Real Yields และความต้องการสินทรัพย์ปลอดภัย",
      "why_not_high": "-",
      "supporting_data": ["DFII10 1.8%", "DTWEXBGS 102.5"],
      "observable_refs": ["obs_005", "obs_006"],
      "source_refs": ["fred_macro.md", "global_snapshot.md"]
    },
    {
      "asset_class": "USD vs THB",
      "asset_bucket": "fx",
      "stance": "OVERWEIGHT",
      "confidence": "MEDIUM",
      "allocation_delta": "+2% vs benchmark",
      "benchmark_ref": "USD/THB",
      "time_horizon": "1-3 months",
      "rationale": "เงินดอลลาร์มีแนวโน้มแข็งค่าเมื่อเทียบกับเงินบาท จากส่วนต่างดอกเบี้ยและฤดูกาลท่องเที่ยวไทยที่ชะลอตัว",
      "why_not_high": "ความไม่แน่นอนของการแทรกแซงจากธนาคารแห่งประเทศไทย",
      "supporting_data": ["USD/THB 36.5", "Policy diff 2.25%"],
      "observable_refs": ["obs_007", "obs_008"],
      "source_refs": ["th_macro.md", "fred_macro.md"]
    },
    {
      "asset_class": "Short-term Liquidity / Cash",
      "asset_bucket": "cash",
      "stance": "UNDERWEIGHT",
      "confidence": "HIGH",
      "allocation_delta": "-2% vs benchmark",
      "benchmark_ref": "3M T-Bill",
      "time_horizon": "1-3 months",
      "rationale": "ลดสัดส่วนเงินสดเพื่อนำไปลงทุนในสินทรัพย์เสี่ยงที่ได้รับประโยชน์จากดอกเบี้ยขาลง",
      "why_not_high": "-",
      "supporting_data": ["3M T-Bill 4.6%"],
      "observable_refs": ["obs_009", "obs_010"],
      "source_refs": ["fred_macro.md", "global_snapshot.md"]
    }
  ],
  "pair_trades": [
    {
      "strategy_name": "US Tech over Europe",
      "instrument_proxy": "Long QQQ / Short VGK ETF",
      "hedge_ratio": "1.0 : 1.0 Notional / Price Ratio",
      "fx_handling": "USD unhedged",
      "entry_trigger": "Price ratio QQQ/VGK drops 2%",
      "stop_loss_trigger": "-3.0% relative spread divergence",
      "target_gain_or_rebalance": "+6.0% spread convergence",
      "max_drawdown_limit": "-4.5% of risk budget",
      "review_frequency": "Weekly",
      "sizing_guidance": "5% of total portfolio",
      "thesis": "หุ้นเทคโนโลยีสหรัฐฯ มีอัตราการเติบโตของกำไรสูงกว่ายุโรปอย่างมีนัยสำคัญ",
      "catalyst": "รายงานผลประกอบการไตรมาสและแนวโน้มการลงทุน AI",
      "risk": "ความเสี่ยงด้านกฎระเบียบเทคโนโลยีและอัตราแลกเปลี่ยน EUR/USD",
      "supporting_data": ["QQQ EPS +15%", "VGK EPS +2%"],
      "observable_refs": ["obs_001", "obs_011"]
    }
  ],
  "risk_scenarios": [
    {
      "scenario_name": "Geopolitical Oil Spike",
      "trigger_to_activate": "Brent Oil > 95.0",
      "trigger_type": "Market Price Threshold",
      "volume_threshold": "Daily trading volume > 50,000 contracts",
      "hedge_size": "5% of portfolio",
      "hedge_purpose": "ป้องกันความเสี่ยงเงินเฟ้อจากราคาพลังงานพุ่งสูงขึ้น",
      "unwind_or_cover_condition": "Brent Oil drops below 85.0"
    }
  ]
}
```
