# Simulador MPC Umidade (Python/Streamlit) — BlockFiel V3 + Slope Theil–Sen (PV_1min)

Este simulador replica o comportamento do bloco **DRYER_HUMIDITY_CTRL_V3_SUPERVISOR_AUTOFF** (Elipse VBA),
incluindo PV Acceptance (QCS irregular), PI preditivo com slope, AutoTune FAST (anti-oscilação),
Supervisor KPI (janela por tempo) e AutoFF tuning.

Além disso, inclui um **bloco virtual de Slope**:
- calcula **PV_1min** = média das amostras QCS dentro de cada minuto
- calcula **slope (%/min)** via **Theil–Sen** usando os **últimos N pontos (N endpoints)** de PV_1min

## Rodar local
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

## Rodar na nuvem (Streamlit Cloud)
1) Suba esta pasta para um repositório GitHub (privado se quiser)
2) No Streamlit Cloud, crie um app apontando para `app.py`
3) O `requirements.txt` instala tudo automaticamente.

## Observação importante
- O simulador usa um modelo de planta **FOPDT** (ajustável) para gerar PV (umidade) a partir de MV (vapor) e distúrbios.
- No real, o bloco recebe PV (QCS) e slope do bloco de tendência.
Aqui, simulamos isso fielmente: PV vem do “QCS virtual” e o slope vem do “bloco slope virtual”.
