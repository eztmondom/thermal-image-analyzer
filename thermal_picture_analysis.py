import cv2
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Helpers
# ----------------------------

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
    print("Thermal cursor estimator (PNG/JPG + scale).")
    #path = input("Add meg a képfájl útvonalát (pl. C:\\kepek\\hokep.png): ").strip().strip('"')
    path = "c:\A\IMG_0045.bmp"
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        print("Nem tudtam beolvasni a képet. Ellenőrizd az útvonalat!")
        return

    # 1) Select scale ROI
    print("\n1) Jelöld ki a SZÍNSKÁLA téglalapját (egérrel), majd ENTER. ESC = kilép.")
    roi = cv2.selectROI("Select SCALE ROI", img, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow("Select SCALE ROI")

    if roi == (0, 0, 0, 0):
        print("ROI nincs kijelölve, kilépek.")
        return

    # 2) Tmin / Tmax
    while True:
        try:
            t_min = float(input("2) Tmin (a skála legalja / legalacsonyabb érték, °C): ").replace(",", "."))
            t_max = float(input("   Tmax (a skála teteje / legmagasabb érték, °C): ").replace(",", "."))
            break
        except ValueError:
            print("Kérlek számot adj meg (pl. 18.5).")

    # 3) Orientation info
    ans = input("3) A skála TETEJE a melegebb (hot)? [y/n] (ha vízszintes skála: BAL oldala hot): ").strip().lower()
    top_is_hot = (ans != "n")

    # Build LUT
    lut_colors, lut_temps = build_lut_from_scale(img, roi, t_min, t_max, top_is_hot=top_is_hot, samples=256)
    print(f"LUT elkészült: {len(lut_temps)} mintapont a skáláról.")
    scale_lab, scale_s, scale_t = build_parametric_scale(lut_colors, lut_temps)
    debug_scale_sampling(img, roi, out_prefix="scalecheck")

    # Interactive window
    win = "Thermal Cursor (ESC=quit)"
    display = img.copy()

    last_xy = None
    pinned_points = []  # list of (x,y,temp)

    def on_mouse(event, x, y, flags, param):
        nonlocal display, last_xy, pinned_points
        if event == cv2.EVENT_MOUSEMOVE:
            last_xy = (x, y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            # pin point
            bgr = img[y, x].tolist()
            temp, d2 = estimate_temp_projected(bgr, scale_lab, scale_s, scale_t)
            pinned_points.append((x, y, temp))
        elif event == cv2.EVENT_RBUTTONDOWN:
            # clear pins
            pinned_points = []

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)

    print("\nKezdheted: mozgasd a kurzort a képen. Bal klikk = pont rögzítése, jobb klikk = pontok törlése, ESC = kilép.\n")

    while True:
        display = img.copy()

        # draw selected ROI for scale
        sx, sy, sw, sh = roi
        cv2.rectangle(display, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 2)
        cv2.putText(display, "SCALE ROI", (sx, max(20, sy - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # show current cursor temp
        if last_xy is not None:
            x, y = last_xy
            if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                bgr = img[y, x].tolist()
                temp, d2 = estimate_temp_projected(bgr, scale_lab, scale_s, scale_t)

                text = f"({x},{y}, {bgr})\n  ~ {temp:.2f} C"
                cv2.circle(display, (x, y), 1, (255, 255, 255), -1)
                cv2.putText(display, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 4)
                cv2.putText(display, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)

        # draw pinned points
        for (px, py, pt) in pinned_points:
            cv2.circle(display, (px, py), 1, (0, 0, 0), -1)
            cv2.circle(display, (px, py), 1, (0, 255, 255), -1)
            cv2.putText(display, f"{pt:.2f}C", (px + 8, py - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3)
            cv2.putText(display, f"{pt:.2f}C", (px + 8, py - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        cv2.imshow(win, display)
        key = cv2.waitKey(15) & 0xFF
        if key == 27:  # ESC
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
