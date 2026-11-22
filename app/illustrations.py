"""Inline SVG illustrations for page banners.

Each function takes the active palette (ui.colors()) and returns an SVG string.
Shapes use the categorical series colors; strokes/text use `currentColor`, so the
illustrations follow the light/dark theme automatically.
"""


def _svg(body: str) -> str:
    return (f'<svg viewBox="0 0 320 200" xmlns="http://www.w3.org/2000/svg" role="img" '
            f'aria-hidden="true" fill="none" stroke-linecap="round" stroke-linejoin="round">{body}</svg>')


def home(c: dict) -> str:
    b, o, a = c["series"]
    bars = "".join(
        f'<rect x="{70 + i * 30}" y="{150 - h}" width="20" height="{h}" rx="4" fill="{b}" opacity="{0.35 + i * 0.13}"/>'
        for i, h in enumerate([34, 52, 70, 96, 118]))
    return _svg(f"""
  <rect x="40" y="18" width="240" height="160" rx="14" stroke="currentColor" stroke-opacity=".25" stroke-width="2"/>
  <line x1="58" y1="150" x2="262" y2="150" stroke="currentColor" stroke-opacity=".3" stroke-width="2"/>
  {bars}
  <path d="M72 128 L110 112 L142 96 L176 66 L214 40" stroke="{o}" stroke-width="3"/>
  <path d="M202 36 L216 39 L212 53" stroke="{o}" stroke-width="3"/>
  <circle cx="256" cy="44" r="24" fill="{a}" opacity=".9"/>
  <text x="256" y="52" text-anchor="middle" font-size="24" font-weight="700" fill="#fff" font-family="sans-serif">$</text>
  <circle cx="58" cy="182" r="5" fill="{b}" opacity=".5"/><circle cx="76" cy="182" r="5" fill="{o}" opacity=".5"/>
  <circle cx="94" cy="182" r="5" fill="{a}" opacity=".5"/>""")


def exploration(c: dict) -> str:
    b, o, a = c["series"]
    heights = [18, 34, 58, 84, 102, 92, 70, 48, 30, 16]
    bars = "".join(f'<rect x="{40 + i * 18}" y="{160 - h}" width="14" height="{h}" rx="3" fill="{b}" opacity=".75"/>'
                   for i, h in enumerate(heights))
    return _svg(f"""
  <line x1="32" y1="160" x2="230" y2="160" stroke="currentColor" stroke-opacity=".3" stroke-width="2"/>
  {bars}
  <circle cx="226" cy="84" r="42" fill="{a}" fill-opacity=".12" stroke="{o}" stroke-width="6"/>
  <line x1="256" y1="114" x2="292" y2="150" stroke="{o}" stroke-width="10"/>
  <path d="M204 92 L216 76 L228 86 L244 66" stroke="{b}" stroke-width="3"/>""")


def statistics(c: dict) -> str:
    b, o, a = c["series"]
    return _svg(f"""
  <line x1="30" y1="160" x2="290" y2="160" stroke="currentColor" stroke-opacity=".3" stroke-width="2"/>
  <path d="M30 158 C 90 156, 110 40, 160 40 C 210 40, 230 156, 290 158" fill="{b}" fill-opacity=".15"
        stroke="{b}" stroke-width="3"/>
  <line x1="160" y1="40" x2="160" y2="160" stroke="{o}" stroke-width="2" stroke-dasharray="5 5"/>
  <path d="M118 160 L118 98" stroke="{a}" stroke-width="2" stroke-dasharray="4 4"/>
  <path d="M202 160 L202 98" stroke="{a}" stroke-width="2" stroke-dasharray="4 4"/>
  <text x="160" y="28" text-anchor="middle" font-size="18" font-weight="700" fill="{o}" font-family="serif">μ</text>
  <text x="250" y="70" font-size="34" font-weight="700" fill="currentColor" fill-opacity=".55" font-family="serif">σ²</text>
  <text x="40" y="70" font-size="26" font-weight="700" fill="currentColor" fill-opacity=".55" font-family="serif">p &lt; α</text>""")


def causal(c: dict) -> str:
    b, o, a = c["series"]
    srcs = [(60, 40), (50, 100), (60, 160)]
    nodes = "".join(f'<circle cx="{x}" cy="{y}" r="16" fill="{b}" opacity=".85"/>' for x, y in srcs)
    edges = "".join(f'<path d="M{x + 18} {y} C 140 {y}, 170 100, 222 100" stroke="currentColor" stroke-opacity=".45" stroke-width="2.5"/>'
                    for x, y in srcs)
    return _svg(f"""
  {edges}
  <path d="M120 40 C 150 60, 150 140, 120 160" stroke="{o}" stroke-width="2.5" stroke-dasharray="6 6"/>
  <path d="M212 92 L224 100 L212 108" stroke="currentColor" stroke-opacity=".6" stroke-width="2.5"/>
  {nodes}
  <rect x="226" y="74" width="66" height="52" rx="12" fill="{a}"/>
  <text x="259" y="108" text-anchor="middle" font-size="24" font-weight="700" fill="#fff" font-family="sans-serif">$</text>
  <text x="146" y="188" text-anchor="middle" font-size="13" fill="currentColor" fill-opacity=".6" font-family="sans-serif">do(X = x)</text>""")


def models(c: dict) -> str:
    b, o, a = c["series"]

    def tree(x: int, y: int, col: str, op: float) -> str:
        return (f'<g opacity="{op}"><path d="M{x} {y} L{x - 26} {y + 34} M{x} {y} L{x + 26} {y + 34} '
                f'M{x - 26} {y + 34} L{x - 40} {y + 66} M{x - 26} {y + 34} L{x - 12} {y + 66} '
                f'M{x + 26} {y + 34} L{x + 40} {y + 66}" stroke="currentColor" stroke-opacity=".45" stroke-width="2"/>'
                f'<circle cx="{x}" cy="{y}" r="9" fill="{col}"/><circle cx="{x - 26}" cy="{y + 34}" r="8" fill="{col}"/>'
                f'<circle cx="{x + 26}" cy="{y + 34}" r="8" fill="{col}"/><circle cx="{x - 40}" cy="{y + 66}" r="6" fill="{col}"/>'
                f'<circle cx="{x - 12}" cy="{y + 66}" r="6" fill="{col}"/><circle cx="{x + 40}" cy="{y + 66}" r="6" fill="{col}"/></g>')
    return _svg(f"""
  {tree(70, 40, b, .55)}{tree(160, 30, o, .8)}{tree(250, 40, a, 1)}
  <text x="115" y="80" text-anchor="middle" font-size="22" font-weight="700" fill="currentColor" fill-opacity=".5" font-family="sans-serif">+</text>
  <text x="205" y="80" text-anchor="middle" font-size="22" font-weight="700" fill="currentColor" fill-opacity=".5" font-family="sans-serif">+</text>
  <rect x="70" y="150" width="180" height="34" rx="17" stroke="currentColor" stroke-opacity=".25" stroke-width="2"/>
  <rect x="74" y="154" width="148" height="26" rx="13" fill="{b}" opacity=".8"/>
  <text x="148" y="172" text-anchor="middle" font-size="13" font-weight="700" fill="#fff" font-family="sans-serif">R² 0,98</text>""")


def explain(c: dict) -> str:
    b, o, a = c["series"]
    steps = [(60, 0, 70, b), (130, 70, 40, b), (170, 110, -24, o), (146, 86, 30, b), (176, 116, -14, o)]
    bars = ""
    for i, (x0, _, w, col) in enumerate(steps):
        y = 30 + i * 26
        x = x0 if w > 0 else x0 + w
        bars += f'<rect x="{x}" y="{y}" width="{abs(w)}" height="18" rx="4" fill="{col}" opacity=".85"/>'
    return _svg(f"""
  <line x1="60" y1="22" x2="60" y2="170" stroke="currentColor" stroke-opacity=".3" stroke-width="2"/>
  {bars}
  <circle cx="252" cy="78" r="34" fill="{a}" fill-opacity=".18" stroke="{a}" stroke-width="4"/>
  <path d="M240 112 L264 112 M242 124 L262 124" stroke="currentColor" stroke-opacity=".6" stroke-width="4"/>
  <path d="M244 82 L252 70 L260 82" stroke="{a}" stroke-width="4"/>
  <path d="M252 70 L252 98" stroke="{a}" stroke-width="4"/>""")


def prediction(c: dict) -> str:
    b, o, a = c["series"]
    return _svg(f"""
  <rect x="36" y="20" width="160" height="164" rx="14" stroke="currentColor" stroke-opacity=".25" stroke-width="2"/>
  <rect x="56" y="42" width="120" height="12" rx="6" fill="currentColor" fill-opacity=".15"/>
  <line x1="56" y1="80" x2="176" y2="80" stroke="currentColor" stroke-opacity=".2" stroke-width="6"/>
  <line x1="56" y1="80" x2="126" y2="80" stroke="{b}" stroke-width="6"/><circle cx="126" cy="80" r="9" fill="{b}"/>
  <line x1="56" y1="112" x2="176" y2="112" stroke="currentColor" stroke-opacity=".2" stroke-width="6"/>
  <line x1="56" y1="112" x2="96" y2="112" stroke="{b}" stroke-width="6"/><circle cx="96" cy="112" r="9" fill="{b}"/>
  <rect x="56" y="140" width="120" height="26" rx="13" fill="{o}"/>
  <path d="M206 102 L234 102" stroke="currentColor" stroke-opacity=".5" stroke-width="3"/>
  <path d="M226 94 L236 102 L226 110" stroke="currentColor" stroke-opacity=".5" stroke-width="3"/>
  <circle cx="270" cy="102" r="36" fill="{a}"/>
  <text x="270" y="112" text-anchor="middle" font-size="30" font-weight="700" fill="#fff" font-family="sans-serif">$</text>
  <path d="M244 150 L296 150" stroke="{a}" stroke-width="3" stroke-opacity=".6"/>
  <path d="M252 162 L288 162" stroke="{a}" stroke-width="3" stroke-opacity=".35"/>""")


def world(c: dict) -> str:
    b, o, a = c["series"]
    pins = "".join(
        f'<path d="M{x} {y} c-9 -14 -9 -24 0 -24 c9 0 9 10 0 24 z" fill="{col}"/>'
        f'<circle cx="{x}" cy="{y - 16}" r="3.5" fill="#fff"/>'
        for x, y, col in [(108, 74, o), (166, 64, a), (196, 120, b), (128, 132, a)])
    return _svg(f"""
  <circle cx="152" cy="100" r="72" fill="{b}" fill-opacity=".12" stroke="currentColor" stroke-opacity=".35" stroke-width="2"/>
  <ellipse cx="152" cy="100" rx="30" ry="72" stroke="currentColor" stroke-opacity=".25" stroke-width="2"/>
  <path d="M80 100 L224 100 M92 64 L212 64 M92 136 L212 136" stroke="currentColor" stroke-opacity=".2" stroke-width="2"/>
  {pins}
  <rect x="236" y="38" width="62" height="34" rx="10" fill="{o}"/>
  <text x="267" y="61" text-anchor="middle" font-size="15" font-weight="700" fill="#fff" font-family="sans-serif">USD</text>
  <rect x="236" y="128" width="62" height="34" rx="10" fill="{a}"/>
  <text x="267" y="151" text-anchor="middle" font-size="15" font-weight="700" fill="#fff" font-family="sans-serif">PPA</text>
  <path d="M267 78 L267 120" stroke="currentColor" stroke-opacity=".5" stroke-width="2.5" stroke-dasharray="4 5"/>
  <path d="M260 112 L267 121 L274 112" stroke="currentColor" stroke-opacity=".5" stroke-width="2.5"/>""")


ART = {"home": home, "exploration": exploration, "statistics": statistics, "causal": causal,
       "models": models, "explain": explain, "prediction": prediction, "world": world}
