import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# ----------------------------
# Helpers
# ----------------------------

def normalize_glyph(bw, size=(32, 32), pad=3):
    """
    bw: 0/255 bináris kép egyetlen karakterrel
    - szoros vágás a tartalomra
    - padding
    - fix méretre húzás
    """

    # biztosítsuk: a "tinta" fehér legyen
    if bw.mean() > 127:
        bw = cv2.bitwise_not(bw)

    ys, xs = np.where(bw > 0)
    if len(xs) == 0 or len(ys) == 0:
        return cv2.resize(bw, size, interpolation=cv2.INTER_NEAREST)

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    glyph = bw[y0:y1+1, x0:x1+1]

    glyph = cv2.copyMakeBorder(glyph, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    glyph = cv2.resize(glyph, size, interpolation=cv2.INTER_NEAREST)
    return glyph

# --- FIX ROIs for 320x240 (IDE írd be a saját számaidat!) ---
SCALE_ROI = (222, 54, 15, 200)
TMAX_ROI = (200, 35, 40, 20)
TMIN_ROI = (200, 257, 40, 20)
TOP_IS_HOT = True  # nálad tipikusan True (felül melegebb)

def draw_rois_debug(img_bgr):
    """
    Draw fixed ROIs on the image for visual verification.
    """
    out = img_bgr.copy()

    rois = [
        (SCALE_ROI, "SCALE"),
        (TMAX_ROI,  "TMAX"),
        (TMIN_ROI,  "TMIN"),
    ]

    for (x, y, w, h), name in rois:
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv2.putText(
            out,
            name,
            (x, max(10, y - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    return out

def preprocess_digit_crop(bgr, name="defname"):
    import cv2
    import numpy as np

    # 1) upscale (OCR-nek és szegmentálásnak sokat számít)
    img = cv2.resize(bgr, None, fx=6, fy=6, interpolation=cv2.INTER_CUBIC)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]

    # 2) fehér számok maszkja (alacsony sat, magas value)
    mask_white = ((S < 60) & (V > 160)).astype(np.uint8) * 255

    # 3) zöld számok/keretek maszkja (ha előfordul)
    # zöld hue kb. 35..95 (OpenCV H: 0..179)
    mask_green = ((H >= 35) & (H <= 95) & (S > 80) & (V > 80)).astype(np.uint8) * 255

    bw = cv2.bitwise_or(mask_white, mask_green)

    # 4) tisztítás
    bw = cv2.medianBlur(bw, 3)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=2)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN,  np.ones((2,2), np.uint8), iterations=1)

    # 5) komponens-szűrés: dobd ki a kis pöttyöket, de a "."-ot hagyd meg
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(bw, connectivity=8)
    Hh, Ww = bw.shape[:2]
    out = np.zeros_like(bw)

    # heurisztikák:
    area_min = int((Hh * Ww) * 0.002)   # általános zajküszöb
    dot_area_min = int((Hh * Ww) * 0.0003)

    for i in range(1, num):
        x, y, w, h, area = stats[i]
        cx, cy = centroids[i]

        # fő karakterek (számjegyek)
        is_main = (area >= area_min) and (h >= Hh * 0.30)

        # pont (kicsi, lent)
        is_dot = (area >= dot_area_min) and (h <= Hh * 0.25) and (cy >= Hh * 0.45)

        # mínusz (lapos, széles komponens)
        # - kicsi magasság
        # - relatíve széles
        # - középtáj / bal oldali régióban szokott lenni
        is_minus = (
                (area >= dot_area_min) and
                (h <= Hh * 0.22) and
                (w >= Ww * 0.07)
        )

        if is_main or is_dot or is_minus:
            out[labels == i] = 255

    out_dir="preprocess_digit_crop"
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(os.path.join(out_dir,f"{name}_dbg_crop.png"), bgr)
    cv2.imwrite(os.path.join(out_dir,f"{name}_dbg_bw.png"), out)  # ahol out a végső bináris
    cv2.imwrite(os.path.join(out_dir,f"{name}_dbg_mask_white.png"), mask_white)
    cv2.imwrite(os.path.join(out_dir,f"{name}_dbg_mask_green.png"), mask_green)
    cv2.imwrite(os.path.join(out_dir,f"{name}_dbg_bw_raw.png"), bw)
    return out

def split_chars(bw):
    if bw is None:
        return []

    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H, W = bw.shape
    boxes = []

    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)

        area = w * h
        if area < 8:  # nagyon pici zaj
            continue

        aspect = w / float(h + 1e-6)

        # számjegy: elég magas
        is_digit = (h >= H * 0.35) and (w >= 5)

        # pont: kicsi és lentebb
        is_dot = (h <= H * 0.28) and (w <= W * 0.18) and (y >= H * 0.40)

        # mínusz: nagyon lapos és szélesebb, mint egy zaj pötty
        # (a te képednél a mínusz rövidebb, ezért W*0.06 körül már engedjük)
        is_minus = (h <= H * 0.20) and (w >= W * 0.06) and (aspect >= 2.2)

        if is_digit or is_dot or is_minus:
            boxes.append((x, y, w, h))

    boxes.sort(key=lambda b: b[0])
    return boxes

def load_templates(folder="templates"):
    mapping = {"minus.png": "-", "dot.png": "."}
    tmpls = {}
    if not os.path.isdir(folder):
        return tmpls

    for fn in os.listdir(folder):
        path = os.path.join(folder, fn)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        _, bw = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if bw.mean() > 180:
            bw = cv2.bitwise_not(bw)

        key = mapping.get(fn)
        if key is None:
            name = os.path.splitext(fn)[0]
            if name.isdigit():
                key = name
        if key is not None:
            bw = normalize_glyph(bw, size=(32, 32))
            tmpls[key] = bw

    # for k, v in tmpls.items():
    #     print("tmpl", k, v.shape, "mean", v.mean())

    out_dir = "loaded_templates"
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(os.path.join(out_dir,"dbg_tmpl_minus.png"), tmpls["-"])
    cv2.imwrite(os.path.join(out_dir,"dbg_tmpl_dot.png"), tmpls["."])
    for i in range(10):
        cv2.imwrite(os.path.join(out_dir,f"dbg_tmpl_{i}.png"), tmpls[str(i)])

    return tmpls

def read_number_from_roi(bgr_roi, templates, min_score=0.60):
    bw = preprocess_digit_crop(bgr_roi)
    boxes = split_chars(bw)
    if not boxes:
        return None

    os.makedirs("read_number_from_roi", exist_ok=True)
    out_dir = "read_number_from_roi"
    out = []
    debug_i = 0
    for x, y, w, h in boxes:
        debug_i = debug_i + 1
        ch = bw[y:y + h, x:x + w]
        # Adjunk egy kis paddinget a karakter köré, hogy a szélek ne zavarják a matchinget
        ch = cv2.copyMakeBorder(ch, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=0)
        ch = normalize_glyph(ch, size=(32, 32))
        cv2.imwrite(os.path.join(out_dir, f"{debug_i}_{y}_dbg_char_norm.png"), ch)

        best_char, best_score = None, -1
        for k, tmpl in templates.items():
            # Normált korrelációs együttható használata az egyszerű különbség helyett
            res = cv2.matchTemplate(ch, tmpl, cv2.TM_CCOEFF_NORMED)
            _, score, _, _ = cv2.minMaxLoc(res)


            # resized = cv2.resize(ch, (tmpl.shape[1], tmpl.shape[0]), interpolation=cv2.INTER_NEAREST)
            # diff = cv2.absdiff(resized, tmpl)
            # score = 1.0 - diff.mean() / 255.0

            # print(f"DBG char: ch={ch[1]}, templ={k}, score={score:.3f}")

            if score > best_score:
                best_score = score
                best_char = k
                # print(f"DBG char: best={best_char} score={best_score:.3f}")

        if best_char is None or best_score < min_score:
            # Ha egy karakter nem biztos, próbáljuk meg a többit, hátha csak egy pont maradt ki
            continue
        out.append(best_char)

    try:
        # print("outtemp:", out)
        if out[0]==".":
                out[0]="-" # todo: hozzáadandó, hogy a szám közepén ha "-" talál javítsa ki "."-re
        return float("".join(out))
    except ValueError:
        return None

def draw_hud(img, lines, origin=(6, 6), font=cv2.FONT_HERSHEY_SIMPLEX,
             font_scale=0.45, thickness=1, pad=6, bg_alpha=0.55):
    """
    Draw a small readable HUD box with multiple text lines.
    Uses semi-transparent dark background for contrast.
    """
    x0, y0 = origin
    # measure text block
    sizes = [cv2.getTextSize(t, font, font_scale, thickness)[0] for t in lines]
    w = max(s[0] for s in sizes) + pad * 2
    h_line = max(s[1] for s in sizes)
    h = (h_line + 4) * len(lines) + pad * 2

    # background overlay
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + w, y0 + h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, bg_alpha, img, 1 - bg_alpha, 0, img)

    # text lines
    y = y0 + pad + h_line
    for t in lines:
        cv2.putText(img, t, (x0 + pad, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        y += (h_line + 4)

def scale_for_display(img, scale=3, interpolation=cv2.INTER_NEAREST):
    """Scale only for display, keeps pixel mapping intact."""
    if scale == 1:
        return img
    return cv2.resize(img, (img.shape[1] * scale, img.shape[0] * scale), interpolation=interpolation)

def debug_scale_sampling(img_bgr, roi, sample_x=None, samples=256, out_prefix="debug"):
    x, y, w, h = roi
    scale = img_bgr[y:y+h, x:x+w].copy()

    # ha nem adsz sample_x-et, középre veszünk (ez csak debug)
    if sample_x is None:
        sample_x = w // 2

    ys = np.linspace(0, h-1, samples).astype(int)
    colors = scale[ys, sample_x, :]  # BGR

    # 1) elmentjük a kivágott ROI-t
    cv2.imwrite(f"{out_prefix}_roi.png", scale)

    # 2) elmentjük a mintavett színcsíkot
    strip = np.zeros((40, samples, 3), dtype=np.uint8)
    strip[:, :, :] = colors[np.newaxis, :, :]
    cv2.imwrite(f"{out_prefix}_strip.png", strip)

    # 3) csatornák görbéje
    b = colors[:, 0].astype(float)
    g = colors[:, 1].astype(float)
    r = colors[:, 2].astype(float)

    plt.figure()
    plt.plot(r, label="R")
    plt.plot(g, label="G")
    plt.plot(b, label="B")
    plt.title("RGB values along the sampled scale line")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_rgb.png", dpi=150)
    plt.show()

    # 4) HSV görbék is sokat segítenek
    hsv = cv2.cvtColor(colors.reshape(-1,1,3), cv2.COLOR_BGR2HSV).reshape(-1,3)
    H, S, V = hsv[:,0].astype(float), hsv[:,1].astype(float), hsv[:,2].astype(float)

    plt.figure()
    plt.plot(H, label="H")
    plt.plot(S, label="S")
    plt.plot(V, label="V")
    plt.title("HSV values along the sampled scale line")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_hsv.png", dpi=150)
    plt.show()
    
def build_parametric_scale(lut_colors_bgr, lut_temps):
    import cv2
    import numpy as np

    lab = cv2.cvtColor(
        lut_colors_bgr.reshape(-1,1,3).astype(np.uint8),
        cv2.COLOR_BGR2LAB
    ).reshape(-1,3).astype(np.float32)

    # cumulative arc-length along the color curve
    diffs = np.diff(lab, axis=0)
    seglen = np.sqrt((diffs**2).sum(axis=1))
    s = np.concatenate([[0.0], np.cumsum(seglen)])
    s /= s[-1]  # normalize to [0..1]

    return lab, s, lut_temps

def build_lut_from_scale(image_bgr, roi, t_min, t_max, top_is_hot=True, samples=256,
                         s_thresh=25, band_half_width=2):
    """
    Robust LUT builder for narrow colorbars inside a bigger ROI.
    - Auto-detects the best column(s) of the colorbar within ROI.
    - Ignores low-saturation pixels (white text/gray UI).
    """
    x, y, w, h = roi
    scale = image_bgr[y:y+h, x:x+w]
    if w <= 0 or h <= 0:
        raise ValueError("Invalid ROI size.")

    hsv = cv2.cvtColor(scale, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # Decide orientation
    vertical = h >= w

    if vertical:
        # Score each column: prefers columns with strong V variation AND decent saturation
        scores = np.zeros(w, dtype=np.float32)
        for j in range(w):
            mask = S[:, j] > s_thresh
            if mask.sum() < h * 0.25:
                continue
            v = V[:, j][mask].astype(np.float32)
            s = S[:, j][mask].astype(np.float32)
            scores[j] = v.std() * (s.mean() + 1.0)

        best_j = int(np.argmax(scores))
        # Define a small band around the best column
        j0 = max(0, best_j - band_half_width)
        j1 = min(w - 1, best_j + band_half_width)

        # Sample along Y; for each Y take median color from the band, using only saturated pixels
        ys = np.linspace(0, h - 1, samples).astype(int)
        colors = []
        for yy in ys:
            band = scale[yy, j0:j1+1, :]  # shape (band, 3)
            band_hsv = hsv[yy, j0:j1+1, :]
            sat_mask = band_hsv[:, 1] > s_thresh
            if np.any(sat_mask):
                bgr_med = np.median(band[sat_mask], axis=0)
            else:
                bgr_med = np.median(band, axis=0)
            colors.append(bgr_med)

        colors = np.array(colors, dtype=np.uint8)

        # Temps mapping
        if top_is_hot:
            temps = np.linspace(t_max, t_min, samples, dtype=np.float32)
        else:
            temps = np.linspace(t_min, t_max, samples, dtype=np.float32)

    else:
        # Horizontal scale (ritkább) – hasonló logika sorokra
        scores = np.zeros(h, dtype=np.float32)
        for i in range(h):
            mask = S[i, :] > s_thresh
            if mask.sum() < w * 0.25:
                continue
            v = V[i, :][mask].astype(np.float32)
            s = S[i, :][mask].astype(np.float32)
            scores[i] = v.std() * (s.mean() + 1.0)

        best_i = int(np.argmax(scores))
        i0 = max(0, best_i - band_half_width)
        i1 = min(h - 1, best_i + band_half_width)

        xs = np.linspace(0, w - 1, samples).astype(int)
        colors = []
        for xx in xs:
            band = scale[i0:i1+1, xx, :]      # shape (band, 3)
            band_hsv = hsv[i0:i1+1, xx, :]    # shape (band, 3)
            sat_mask = band_hsv[:, 1] > s_thresh
            if np.any(sat_mask):
                bgr_med = np.median(band[sat_mask], axis=0)
            else:
                bgr_med = np.median(band, axis=0)
            colors.append(bgr_med)

        colors = np.array(colors, dtype=np.uint8)

        if top_is_hot:  # itt “left_is_hot”-ként értelmezd
            temps = np.linspace(t_max, t_min, samples, dtype=np.float32)
        else:
            temps = np.linspace(t_min, t_max, samples, dtype=np.float32)

    # Deduplicate very similar consecutive colors (optional, helps with banding)
    kept_c = [colors[0]]
    kept_t = [temps[0]]
    for c, t in zip(colors[1:], temps[1:]):
        if np.sum((c.astype(int) - kept_c[-1].astype(int)) ** 2) > 2:
            kept_c.append(c)
            kept_t.append(t)

    return np.array(kept_c, dtype=np.uint8), np.array(kept_t, dtype=np.float32)


def estimate_temp_projected(bgr, scale_lab, scale_s, scale_t):
    import cv2
    import numpy as np

    pix_lab = cv2.cvtColor(
        np.array([[bgr]], dtype=np.uint8),
        cv2.COLOR_BGR2LAB
    )[0,0].astype(np.float32)

    # find closest point on the curve
    d2 = np.sum((scale_lab - pix_lab)**2, axis=1)
    idx = int(np.argmin(d2))

    s = scale_s[idx]
    temp = np.interp(s, scale_s, scale_t)
    return float(temp), float(d2[idx])


# ----------------------------
# Main
# ----------------------------

def main():

    DISPLAY_SCALE = 3  # 2 vagy 3 ajánlott 320x240-hez
    HUD_FONT = cv2.FONT_HERSHEY_SIMPLEX
    HUD_SCALE = 0.3  # betűméret (kicsi)
    HUD_THICK = 1  # vékony
    HUD_PAD = 6  # belső margó
    HUD_BG_ALPHA = 0.55  # félig átlátszó háttér


    print("Thermal cursor estimator (PNG/JPG + scale).")
    #path = input("Add meg a képfájl útvonalát (pl. C:\\kepek\\hokep.png): ").strip().strip('"')
    path = r"c:\A\IMG_0045.bmp"
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        print("Nem tudtam beolvasni a képet. Ellenőrizd az útvonalat!")
        return

    # 1) Select scale ROI
    # print("\n1) Jelöld ki a SZÍNSKÁLA téglalapját (egérrel), majd ENTER. ESC = kilép.")
    # roi = cv2.selectROI("Select SCALE ROI", img, showCrosshair=True, fromCenter=False)
    # cv2.destroyWindow("Select SCALE ROI")
    #
    # if roi == (0, 0, 0, 0):
    #     print("ROI nincs kijelölve, kilépek.")
    #     return
    #
    # # 2) Tmin / Tmax
    # while True:
    #     try:
    #         t_min = float(input("2) Tmin (a skála legalja / legalacsonyabb érték, °C): ").replace(",", "."))
    #         t_max = float(input("   Tmax (a skála teteje / legmagasabb érték, °C): ").replace(",", "."))
    #         break
    #     except ValueError:
    #         print("Kérlek számot adj meg (pl. 18.5).")
    #
    # # 3) Orientation info
    # ans = input("3) A skála TETEJE a melegebb (hot)? [y/n] (ha vízszintes skála: BAL oldala hot): ").strip().lower()
    # top_is_hot = (ans != "n")

    def crop_roi(img, roi):
        x, y, w, h = roi
        return img[y:y + h, x:x + w].copy()

    # 1) FIX ROI használata
    roi = SCALE_ROI
    top_is_hot = TOP_IS_HOT

    # 2) Template-ek betöltése
    templates = load_templates("templates")
    print("TEMPLATE KEYS:", sorted(templates.keys()))
    print("TEMPLATE COUNT:", len(templates))
    if not templates:
        print("Nincs templates mappa vagy üres. Kézzel kell Tmin/Tmax.")
        t_min = float(input("Tmin (°C): ").replace(",", "."))
        t_max = float(input("Tmax (°C): ").replace(",", "."))
    else:
        print("=== TMAX ===")
        t_max = read_number_from_roi(crop_roi(img, TMAX_ROI), templates, min_score=0.40)
        print("=== TMIN ===")
        t_min = read_number_from_roi(crop_roi(img, TMIN_ROI), templates, min_score=0.40)

        dump_ocr_debug(img, TMAX_ROI, templates, name="TMAX")
        dump_ocr_debug(img, TMIN_ROI, templates, name="TMIN")
        print("Debug képek mentve: ./ocr_debug mappába")

        if t_min is None or t_max is None:
            print("Template OCR nem sikerült biztosan. Fallback kézi beírás.")
            #t_min = float(input("Tmin (°C): ").replace(",", "."))
            t_min=float("-4.8")
            #t_max = float(input("Tmax (°C): ").replace(",", "."))
            t_max=float("2.8")

        else:
            print(f"Auto Tmin/Tmax: Tmin={t_min}, Tmax={t_max}")

    # Build LUT
    lut_colors, lut_temps = build_lut_from_scale(img, roi, t_min, t_max, top_is_hot=top_is_hot, samples=256)
    print(f"LUT elkészült: {len(lut_temps)} mintapont a skáláról.")
    scale_lab, scale_s, scale_t = build_parametric_scale(lut_colors, lut_temps)
    # debug_scale_sampling(img, roi, out_prefix="scalecheck")

    # Interactive window
    win = "Thermal Cursor (ESC=quit)"
    display = img.copy()

    last_xy = None
    pinned_points = []  # list of (x,y,temp)

    def on_mouse(event, x, y, flags, param):
        nonlocal last_xy, pinned_points
        # map display coords -> original coords
        ox = int(x / DISPLAY_SCALE)
        oy = int(y / DISPLAY_SCALE)

        if event == cv2.EVENT_MOUSEMOVE:
            last_xy = (ox, oy)
        elif event == cv2.EVENT_LBUTTONDOWN:
            if 0 <= ox < img.shape[1] and 0 <= oy < img.shape[0]:
                bgr = img[oy, ox].tolist()
                temp, d2 = estimate_temp_projected(bgr, scale_lab, scale_s, scale_t)
                pinned_points.append((ox, oy, temp))
        elif event == cv2.EVENT_RBUTTONDOWN:
            pinned_points = []

    #cv2.namedWindow(win, cv2.WINDOW_NORMAL) #vissza
    #cv2.setMouseCallback(win, on_mouse)

    print("\nKezdheted: mozgasd a kurzort a képen. Bal klikk = pont rögzítése, jobb klikk = pontok törlése, ESC = kilép.\n")

    # dbg = draw_rois_debug(img)
    # cv2.imshow("ROI check", dbg)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    while True:
        display = img.copy()

        # draw selected ROI for scale
        # sx, sy, sw, sh = roi
        # cv2.rectangle(display, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 2)
        # cv2.putText(display, "SCALE ROI", (sx, max(20, sy - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # show current cursor temp
        # draw pinned points
        for (px, py, pt) in pinned_points:
            cv2.circle(display, (px, py), 1, (0, 0, 0), -1)
            cv2.circle(display, (px, py), 1, (0, 255, 255), -1)
            cv2.putText(display, f"{pt:.2f}C", (px + 8, py - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3)
            cv2.putText(display, f"{pt:.2f}C", (px + 8, py - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        # cv2.imshow(win, display)
        lines = []
        if last_xy is not None:
            x, y = last_xy
            if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                bgr = img[y, x].tolist()
                temp, d2 = estimate_temp_projected(bgr, scale_lab, scale_s, scale_t)
                lines.append(f"XY: {x},{y}")
                lines.append(f"T:  {temp:.2f} C")

        if len(pinned_points) > 0:
            lines.append(f"Pins: {len(pinned_points)} (RMB clear)")

        if not lines:
            lines = ["Move mouse to read temperature", "LMB pin, RMB clear, ESC quit"]

        draw_hud(display, lines, origin=(6, 6),
                 font=HUD_FONT, font_scale=HUD_SCALE, thickness=HUD_THICK,
                 pad=HUD_PAD, bg_alpha=HUD_BG_ALPHA)

        # finally show scaled-up view
        # view = scale_for_display(display, scale=DISPLAY_SCALE, interpolation=cv2.INTER_NEAREST)
        # cv2.imshow(win, view)

        break
        key = cv2.waitKey(15) & 0xFF
        if key == 27:  # ESC
            break

    cv2.destroyAllWindows()


def dump_ocr_debug(img_bgr, roi, templates, name="TMAX", out_dir="ocr_debug", min_score=0.60):
    os.makedirs(out_dir, exist_ok=True)

    x,y,w,h = roi
    crop = img_bgr[y:y+h, x:x+w].copy()
    cv2.imwrite(os.path.join(out_dir, f"{name}_crop.png"), crop)

    bw = preprocess_digit_crop(crop, name)
    cv2.imwrite(os.path.join(out_dir, f"{name}_bw.png"), bw)

    boxes = split_chars(bw)

    # dobozok rajza
    vis = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
    for (bx,by,bw_,bh_) in boxes:
        cv2.rectangle(vis, (bx,by), (bx+bw_,by+bh_), (0,255,0), 1)
    cv2.imwrite(os.path.join(out_dir, f"{name}_boxes.png"), vis)

    # karakterenként mentés + score lista
    lines = []
    for i,(bx,by,bw_,bh_) in enumerate(boxes):
        ch = bw[by:by+bh_, bx:bx+bw_]
        ch = cv2.copyMakeBorder(ch, 2,2,2,2, cv2.BORDER_CONSTANT, value=0)
        cv2.imwrite(os.path.join(out_dir, f"{name}_char_{i}.png"), ch)

        best_char, best_score = None, -1
        best2_char, best2_score = None, -1

        for k, tmpl in templates.items():
            resized = cv2.resize(ch, (tmpl.shape[1], tmpl.shape[0]), interpolation=cv2.INTER_NEAREST)
            diff = cv2.absdiff(resized, tmpl)
            score = 1.0 - diff.mean()/255.0

            if score > best_score:
                best2_char, best2_score = best_char, best_score
                best_char, best_score = k, score
            elif score > best2_score:
                best2_char, best2_score = k, score

        ok = "OK" if best_score >= min_score else "FAIL"
        lines.append(f"{name} char{i}: best={best_char} {best_score:.3f} ({ok}), 2nd={best2_char} {best2_score:.3f}")

    with open(os.path.join(out_dir, f"{name}_scores.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return boxes

if __name__ == "__main__":
    main()
