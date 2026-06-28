"""
menu.py
=======
Pantalla de selección de ejercicio usando tkinter.

Interfaz pública
----------------
  from menu import mostrar_menu

  ejercicio = mostrar_menu(ejercicio_actual="sentadilla")
  # Retorna el nombre del ejercicio elegido, o None si se cierra con ESC.
"""

import tkinter as tk
from tkinter import font as tkfont
from angulos_referencia import EJERCICIOS

EJERCICIO_KEYS = list(EJERCICIOS.keys())

# Paleta — fondo claro, texto oscuro
BG        = "#f0f4ff"
BG_BTN    = "#ffffff"
BG_SEL    = "#dde8ff"
FG_BTN    = "#0a1a3a"
FG_SEL    = "#0a1a3a"
BRD_BTN   = "#c0ccee"
BRD_SEL   = "#2255dd"
BG_START  = "#2255dd"
FG_START  = "#ffffff"
FG_DIM    = "#8899bb"
FG_SUB    = "#3355aa"
FG_TITLE  = "#0a1a3a"


def mostrar_menu(ejercicio_actual: str = "sentadilla") -> str | None:
    """
    Muestra una ventana para elegir el ejercicio.

    Retorna el nombre del ejercicio elegido, o None si se canceló.
    """
    elegido   = [ejercicio_actual]
    cancelado = [False]

    root = tk.Tk()
    root.title("Pose Analysis")
    root.configure(bg=BG)
    root.resizable(False, False)

    W, H = 480, 520
    root.update_idletasks()
    x = (root.winfo_screenwidth()  - W) // 2
    y = (root.winfo_screenheight() - H) // 2
    root.geometry(f"{W}x{H}+{x}+{y}")

    # Fuentes
    try:
        f_titulo = tkfont.Font(family="Helvetica Neue", size=22, weight="bold")
        f_sub    = tkfont.Font(family="Helvetica Neue", size=11)
        f_boton  = tkfont.Font(family="Helvetica Neue", size=13, weight="bold")
        f_tecla  = tkfont.Font(family="Helvetica Neue", size=10)
    except Exception:
        f_titulo = tkfont.Font(size=22, weight="bold")
        f_sub    = tkfont.Font(size=11)
        f_boton  = tkfont.Font(size=13, weight="bold")
        f_tecla  = tkfont.Font(size=10)

    # Título
    tk.Label(root, text="POSE ANALYSIS",
             font=f_titulo, bg=BG, fg=FG_TITLE).pack(pady=(32, 2))
    tk.Label(root, text="Elegí el ejercicio para analizar",
             font=f_sub, bg=BG, fg=FG_SUB).pack(pady=(0, 24))

    # Botones
    frame_botones = tk.Frame(root, bg=BG)
    frame_botones.pack(fill="x", padx=40)
    teclas = ["1", "2", "3", "4", "5"]

    for i, (key, cfg) in enumerate(EJERCICIOS.items()):
        es_actual = (key == ejercicio_actual)
        bg_b  = BG_SEL  if es_actual else BG_BTN
        fg_b  = FG_SEL  if es_actual else FG_BTN
        brd_b = BRD_SEL if es_actual else BRD_BTN

        def hacer_click(k=key):
            elegido[0] = k
            root.destroy()

        btn = tk.Button(
            frame_botones,
            text=f"  [{teclas[i]}]  {cfg['descripcion']}",
            font=f_boton,
            bg=bg_b, fg=fg_b,
            activebackground="#c8d8ff",
            activeforeground="#0a1a3a",
            relief="flat", bd=0,
            highlightthickness=1,
            highlightbackground=brd_b,
            anchor="w", padx=16, pady=12,
            command=hacer_click,
            cursor="hand2",
        )
        btn.pack(fill="x", pady=5)
        btn.bind("<Enter>", lambda e, b=btn: b.configure(bg="#e8eeff"))
        btn.bind("<Leave>", lambda e, b=btn, bg=bg_b: b.configure(bg=bg))

    # Teclas de acceso rápido
    def on_key(event):
        if event.char in teclas:
            idx = int(event.char) - 1
            if idx < len(EJERCICIO_KEYS):
                elegido[0] = EJERCICIO_KEYS[idx]
                root.destroy()
        elif event.keysym in ("Return", "space"):
            root.destroy()
        elif event.keysym == "Escape":
            cancelado[0] = True
            root.destroy()

    root.bind("<Key>", on_key)
    root.protocol("WM_DELETE_WINDOW",
                  lambda: [cancelado.__setitem__(0, True), root.destroy()])

    # Botón comenzar
    tk.Frame(root, bg=BG, height=10).pack()
    tk.Button(
        root,
        text="▶  Comenzar",
        font=f_boton,
        bg=BG_START, fg=FG_START,
        activebackground="#1a44bb",
        activeforeground="#ffffff",
        relief="flat", bd=0,
        padx=24, pady=10,
        command=root.destroy,
        cursor="hand2",
    ).pack(pady=(10, 4))

    tk.Label(root,
             text="ESC para cancelar   •   Enter para confirmar",
             font=f_tecla, bg=BG, fg=FG_DIM).pack(pady=(8, 0))

    root.mainloop()

    return None if cancelado[0] else elegido[0]