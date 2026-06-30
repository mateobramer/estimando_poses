"""
menu.py
=======
Pantalla de selección de ejercicio usando tkinter.
Lee los ejercicios dinámicamente desde angulos_referencia.EJERCICIOS.
Para agregar un ejercicio al menú: solo agregarlo en angulos_referencia.py.

Interfaz pública
----------------
  from menu import mostrar_menu
  ejercicio = mostrar_menu(ejercicio_actual="media_sentadilla")
  # Retorna el nombre del ejercicio elegido, o None si se cierra con ESC.
"""

import tkinter as tk
from tkinter import font as tkfont
from angulos_referencia import EJERCICIOS

# ---------------------------------------------------------------------------
# Agrupación de ejercicios — editar acá si se agregan nuevos
# ---------------------------------------------------------------------------

GRUPOS = {
    "Piernas": [
        "media_sentadilla",
        "sentadilla_profunda",
        "sentadilla",
        "jump_squat",
        "peso_muerto_una_pierna",
        "estocada",
        "patada_lateral",
    ],
    "Cuerpo completo": [
        "jumping_jacks",
        "flexion_brazos",
    ],
    "Hombros": [
        "press_hombros",
    ],
}

# Emojis por ejercicio
ICONOS = {
    "media_sentadilla":       "🏋️",
    "sentadilla_profunda":    "🏋️",
    "sentadilla":             "🏋️",
    "jump_squat":             "🦘",
    "peso_muerto_una_pierna": "🦵",
    "estocada":               "🚶",
    "patada_lateral":         "🥋",
    "jumping_jacks":          "⭐",
    "flexion_brazos":         "💪",
    "press_hombros":          "🏅",
}

# Paleta — fondo blanco, acento azul
BG          = "#ffffff"
BG_CARD     = "#f7f7f7"
BG_CARD_HOV = "#f0f0f0"
BG_SEL      = "#edf2ff"
FG_PRIMARY  = "#0a0a0a"
FG_MUTED    = "#999999"
BRD_DEFAULT = "#e0e0e0"
BRD_SEL     = "#2255dd"
ACCENT      = "#2255dd"
ACCENT_TXT  = "#ffffff"


def mostrar_menu(ejercicio_actual: str | None = None) -> str | None:
    """
    Muestra una ventana para elegir el ejercicio.
    Retorna el nombre del ejercicio elegido, o None si se canceló.
    """
    # Ejercicios disponibles (solo los que existen en EJERCICIOS)
    ejercicios_disponibles = list(EJERCICIOS.keys())
    if ejercicio_actual not in ejercicios_disponibles:
        ejercicio_actual = ejercicios_disponibles[0] if ejercicios_disponibles else None

    elegido   = [ejercicio_actual]
    cancelado = [False]

    root = tk.Tk()
    root.title("Pose Analysis")
    root.configure(bg=BG)
    root.resizable(False, False)

    W = 500
    root.update_idletasks()
    x = (root.winfo_screenwidth()  - W) // 2
    y = max(0, (root.winfo_screenheight() - 600) // 2)
    root.geometry(f"{W}x600+{x}+{y}")


    f_app    = tkfont.Font(family="Helvetica Neue", size=10, weight="bold")
    f_titulo = tkfont.Font(family="Helvetica Neue", size=26, weight="bold")
    f_grupo  = tkfont.Font(family="Helvetica Neue", size=10, weight="bold")
    f_nombre = tkfont.Font(family="Helvetica Neue", size=12, weight="bold")
    f_meta   = tkfont.Font(family="Helvetica Neue", size=10)
    f_hint   = tkfont.Font(family="Helvetica Neue", size=10)
    f_btn    = tkfont.Font(family="Helvetica Neue", size=13, weight="bold")

    canvas = tk.Canvas(root, bg=BG, highlightthickness=0)
    scrollbar = tk.Scrollbar(root, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=scrollbar.set)
    scrollbar.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)

    frame = tk.Frame(canvas, bg=BG)
    canvas_window = canvas.create_window((0, 0), window=frame, anchor="nw")

    def on_frame_configure(e):
        canvas.configure(scrollregion=canvas.bbox("all"))

    def on_canvas_configure(e):
        canvas.itemconfig(canvas_window, width=e.width)

    frame.bind("<Configure>", on_frame_configure)
    canvas.bind("<Configure>", on_canvas_configure)

    # Mousewheel
    root.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(-1*(e.delta//120), "units"))
    root.bind_all("<Button-4>",   lambda e: canvas.yview_scroll(-1, "units"))
    root.bind_all("<Button-5>",   lambda e: canvas.yview_scroll(1, "units"))

    pad = tk.Frame(frame, bg=BG)
    pad.pack(fill="x", padx=24, pady=(24, 0))

    # Encabezado
    tk.Label(pad, text="POSE ANALYSIS", font=f_app,
             bg=BG, fg=FG_MUTED, anchor="w").pack(fill="x")
    tk.Label(pad, text="ELEGÍ UN EJERCICIO", font=f_titulo,
             bg=BG, fg=FG_PRIMARY, anchor="w").pack(fill="x", pady=(2, 16))

    # Referencia a todos los botones para poder deseleccionar
    todos_los_btns: list[tuple[str, tk.Frame, tk.Label, tk.Label]] = []

    def seleccionar(key: str):
        elegido[0] = key
        for k, card, lbl_nombre, lbl_meta in todos_los_btns:
            if k == key:
                card.configure(bg=BG_SEL, highlightbackground=BRD_SEL, highlightthickness=2)
                lbl_nombre.configure(bg=BG_SEL, fg=ACCENT)
                lbl_meta.configure(bg=BG_SEL)
            else:
                card.configure(bg=BG_CARD, highlightbackground=BRD_DEFAULT, highlightthickness=1)
                lbl_nombre.configure(bg=BG_CARD, fg=FG_PRIMARY)
                lbl_meta.configure(bg=BG_CARD)

    # Generar grupos
    numero = 1
    tecla_a_key: dict[str, str] = {}

    for grupo_nombre, claves in GRUPOS.items():
        claves_visibles = [k for k in claves if k in EJERCICIOS]
        if not claves_visibles:
            continue

        # Línea separadora + label de grupo
        sep_frame = tk.Frame(pad, bg=BG)
        sep_frame.pack(fill="x", pady=(8, 4))
        tk.Frame(sep_frame, bg=BRD_DEFAULT, height=1).pack(fill="x", pady=(0, 6))
        tk.Label(sep_frame, text=grupo_nombre.upper(), font=f_grupo,
                 bg=BG, fg=FG_MUTED, anchor="w").pack(fill="x")

        # Grilla de 2 columnas
        grid = tk.Frame(pad, bg=BG)
        grid.pack(fill="x", pady=(0, 4))
        grid.columnconfigure(0, weight=1)
        grid.columnconfigure(1, weight=1)

        col, row = 0, 0
        for key in claves_visibles:
            cfg    = EJERCICIOS[key]
            desc   = cfg.get("descripcion", key)
            camara = cfg.get("camara", "").replace("_o_", " o ").replace("_", " ")
            icono  = ICONOS.get(key, "●")
            tecla  = str(numero)
            tecla_a_key[tecla] = key
            numero += 1

            es_sel = (key == elegido[0])
            bg_c   = BG_SEL    if es_sel else BG_CARD
            brd    = BRD_SEL   if es_sel else BRD_DEFAULT
            brd_w  = 2         if es_sel else 1
            fg_n   = ACCENT    if es_sel else FG_PRIMARY

            card = tk.Frame(grid, bg=bg_c,
                            highlightbackground=brd,
                            highlightthickness=brd_w,
                            cursor="hand2")
            card.grid(row=row, column=col, sticky="ew", padx=(0 if col==0 else 4, 0), pady=4)

            inner = tk.Frame(card, bg=bg_c)
            inner.pack(fill="x", padx=12, pady=10)

            # Fila: icono + textos
            tk.Label(inner, text=icono, font=tkfont.Font(size=18),
                     bg=bg_c).pack(side="left", padx=(0, 10))

            text_frame = tk.Frame(inner, bg=bg_c)
            text_frame.pack(side="left", fill="x", expand=True)

            lbl_nombre = tk.Label(text_frame, text=desc.upper(), font=f_nombre,
                                  bg=bg_c, fg=fg_n, anchor="w")
            lbl_nombre.pack(fill="x")

            meta_txt = camara if camara else f"[{tecla}]"
            if camara:
                meta_txt = f"[{tecla}]  {camara}"
            lbl_meta = tk.Label(text_frame, text=meta_txt, font=f_meta,
                                bg=bg_c, fg=FG_MUTED, anchor="w")
            lbl_meta.pack(fill="x")

            todos_los_btns.append((key, card, lbl_nombre, lbl_meta))

            def hacer_click(k=key):
                seleccionar(k)

            for widget in [card, inner, lbl_nombre, lbl_meta]:
                widget.bind("<Button-1>", lambda e, k=key: hacer_click(k))

            # Hover
            def on_enter(e, c=card, k=key, ln=lbl_nombre, lm=lbl_meta):
                if elegido[0] != k:
                    c.configure(bg=BG_CARD_HOV)
                    ln.configure(bg=BG_CARD_HOV)
                    lm.configure(bg=BG_CARD_HOV)

            def on_leave(e, c=card, k=key, ln=lbl_nombre, lm=lbl_meta):
                if elegido[0] != k:
                    c.configure(bg=BG_CARD)
                    ln.configure(bg=BG_CARD)
                    lm.configure(bg=BG_CARD)

            for widget in [card, inner, lbl_nombre, lbl_meta]:
                widget.bind("<Enter>", on_enter)
                widget.bind("<Leave>", on_leave)

            col += 1
            if col > 1:
                col = 0
                row += 1

    # Footer
    tk.Frame(pad, bg=BRD_DEFAULT, height=1).pack(fill="x", pady=(16, 12))

    footer = tk.Frame(pad, bg=BG)
    footer.pack(fill="x", pady=(0, 24))

    tk.Label(footer, text=f"Teclas 1–{len(tecla_a_key)} para elegir · ESC para salir",
             font=f_hint, bg=BG, fg=FG_MUTED).pack(side="left")

    def comenzar():
        root.destroy()

    btn = tk.Button(
        footer,
        text="▶  COMENZAR",
        font=f_btn,
        bg=ACCENT, fg=ACCENT_TXT,
        activebackground="#1a44bb",
        activeforeground=ACCENT_TXT,
        relief="flat", bd=0,
        padx=20, pady=8,
        command=comenzar,
        cursor="hand2",
    )
    btn.pack(side="right")

    # Teclas de acceso rápido
    def on_key(event):
        if event.char in tecla_a_key:
            seleccionar(tecla_a_key[event.char])
        elif event.keysym in ("Return", "space"):
            root.destroy()
        elif event.keysym == "Escape":
            cancelado[0] = True
            root.destroy()

    root.bind("<Key>", on_key)
    root.protocol("WM_DELETE_WINDOW",
                  lambda: [cancelado.__setitem__(0, True), root.destroy()])

    root.mainloop()
    return None if cancelado[0] else elegido[0]