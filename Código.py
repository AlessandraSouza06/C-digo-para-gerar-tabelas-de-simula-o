"""
Gera: cenários (Tabela 2), Monte Carlo (Tabela 3) e sensibilidade (Figura 4).
"""
import numpy as np

# ---------------- Parâmetros (Tabela 1): base, mínimo, máximo ----------------
# chave: (base, min, max)
FAIXAS = {
    "pedidos_dia": (400, 250, 600),      # pedidos por dia útil
    "dias_mes":    (22, 22, 22),
    "f":           (0.20, 0.10, 0.30),   # fração por telefone (fora do escopo)
    "tm":          (6.0, 4.0, 9.0),      # tempo manual por pedido (min)
    "em":          (0.03, 0.02, 0.05),   # erro no registro manual
    "ce":          (60, 30, 100),        # custo por pedido com erro (R$)
    "ch":          (30, 25, 38),         # custo-hora carregado (R$)
    "s":           (0.70, 0.55, 0.85),   # automatizáveis sem intervenção
    "tr":          (2.0, 1.5, 3.0),      # tempo de revisão por exceção (min)
    "ea":          (0.008, 0.003, 0.015),# erro residual (automatizados)
    "I":           (120_000, 80_000, 180_000),  # investimento inicial (R$)
    "Crec":        (12_000, 8_000, 16_000),     # custo recorrente mensal (R$)
}
BASE = {k: v[0] for k, v in FAIXAS.items()}

CONSERVADOR = dict(BASE, tm=5.0, em=0.025, ce=45, ch=27, s=0.55, tr=2.0,
                   ea=0.012, I=160_000, Crec=14_000)
OTIMISTA = dict(BASE, tm=7.0, em=0.04, ce=80, ch=33, s=0.85, tr=2.0,
                ea=0.005, I=100_000, Crec=10_000)


# ---------------- Modelo (seção 3.3) ----------------
def modelo(p):
    """Aceita dict de escalares ou de arrays numpy (vetorizado)."""
    N = p["pedidos_dia"] * p["dias_mes"]
    f, s, tm, tr = p["f"], p["s"], p["tm"], p["tr"]
    H_antes = N * tm / 60
    H_depois = (N * f * tm + N * (1 - f) * (1 - s) * tr) / 60
    E_antes = N * p["em"]
    E_depois = N * f * p["em"] + N * (1 - f) * p["ea"]
    economia = ((H_antes - H_depois) * p["ch"]
                + (E_antes - E_depois) * p["ce"] - p["Crec"])
    with np.errstate(divide="ignore", invalid="ignore"):
        payback = np.where(economia > 0, p["I"] / economia, np.inf)
        roi24 = (24 * economia - p["I"]) / p["I"]
    return dict(N=N, H_antes=H_antes, H_depois=H_depois,
                min_pedido_antes=H_antes * 60 / N, min_pedido_depois=H_depois * 60 / N,
                E_antes=E_antes, E_depois=E_depois,
                taxa_erro_antes=E_antes / N, taxa_erro_depois=E_depois / N,
                custo_antes=H_antes * p["ch"] + E_antes * p["ce"],
                custo_depois=H_depois * p["ch"] + E_depois * p["ce"] + p["Crec"],
                economia=economia, payback=payback, roi24=roi24)


def tabela2():
    print("\n=== Cenários (Tabela 2) ===")
    for nome, p in [("Base", BASE), ("Conservador", CONSERVADOR), ("Otimista", OTIMISTA)]:
        r = modelo(p)
        print(f"{nome:12s} horas {r['H_antes']:.0f}->{r['H_depois']:.0f} | "
              f"min/pedido {r['min_pedido_antes']:.1f}->{r['min_pedido_depois']:.1f} | "
              f"erro {r['taxa_erro_antes']:.1%}->{r['taxa_erro_depois']:.1%} | "
              f"custo R$ {r['custo_antes']:,.0f}->R$ {r['custo_depois']:,.0f} | "
              f"economia R$ {r['economia']:,.0f} | payback {r['payback']:.1f} m | "
              f"ROI24 {r['roi24']:.0%}")


def monte_carlo(n=10_000, seed=42):
    """Distribuição triangular: mínimo, moda (base) e máximo da faixa."""
    rng = np.random.default_rng(seed)
    amostra = {}
    for k, (base, lo, hi) in FAIXAS.items():
        amostra[k] = np.full(n, float(base)) if lo == hi else rng.triangular(lo, base, hi, n)
    r = modelo(amostra)
    pb, ec, roi = r["payback"], r["economia"], r["roi24"]
    print("\n=== Monte Carlo (Tabela 3) ===")
    print(f"{'':16s}{'P10':>10s}{'Mediana':>10s}{'P90':>10s}")
    for nome, v, fmt in [("Economia (R$)", ec, "{:10,.0f}"), ("Payback (meses)", pb, "{:10.1f}"),
                         ("ROI 24 meses", roi, "{:10.0%}")]:
        p10, p50, p90 = np.percentile(v, [10, 50, 90])
        print(f"{nome:16s}" + fmt.format(p10) + fmt.format(p50) + fmt.format(p90))
    print(f"P(payback <= 12 m) = {np.mean(pb <= 12):.1%} | P(<= 24 m) = {np.mean(pb <= 24):.1%} "
          f"| P(economia > 0) = {np.mean(ec > 0):.2%}")


def sensibilidade():
    """Varia um parâmetro por vez entre os extremos da faixa (Figura 4)."""
    print("\n=== Sensibilidade do payback (meses) ===")
    linhas = []
    for k, (base, lo, hi) in FAIXAS.items():
        if lo == hi:
            continue
        a = modelo(dict(BASE, **{k: lo}))["payback"]
        b = modelo(dict(BASE, **{k: hi}))["payback"]
        linhas.append((k, lo, hi, float(a), float(b)))
    for k, lo, hi, a, b in sorted(linhas, key=lambda x: -abs(x[3] - x[4])):
        print(f"{k:12s} min={lo:<10} -> {a:6.1f} | max={hi:<10} -> {b:6.1f}")


def main():
    tabela2()
    monte_carlo()
    sensibilidade()


if __name__ == "__main__":
    main()
