import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageDraw, ImageOps, ImageTk
import numpy as np
import tensorflow as tf
import os

MODEL_PATH = "mnist_cnn_custom.h5"
CANVAS_W, CANVAS_H = 300, 300
PEN = 5 # Độ đậm của bút

# ================= XỬ LÝ ẢNH (GIỮ NGUYÊN) =================
def preprocess(img, size=(28,28)):
    img = img.convert("L")
    arr = np.array(img)

    # Nếu nền sáng (mean > 127) thì đảo để nền tối, nét sáng
    if arr.mean() > 127:
        img = ImageOps.invert(img)
        arr = np.array(img)

    mask = arr > 20
    if not mask.any():
        return np.zeros((28,28,1))

    ys, xs = np.where(mask)
    img = img.crop((xs.min(), ys.min(), xs.max()+1, ys.max()+1))

    w, h = img.size
    if w > h:
        img = img.resize((20, int(20*h/w)))
    else:
        img = img.resize((int(20*w/h), 20))

    canvas = Image.new("L", (28,28))
    canvas.paste(img, ((28-img.width)//2, (28-img.height)//2))

    arr = np.array(canvas)/255.0
    return np.expand_dims(arr, -1)


# ================= LOAD MODEL =================
if not os.path.exists(MODEL_PATH):
    messagebox.showerror("Lỗi", "Không tìm thấy model!")
    exit()

model = tf.keras.models.load_model(MODEL_PATH)


# ================== BIẾN TOÀN CỤC ==================
img = Image.new("L", (CANVAS_W, CANVAS_H), 255)
draw = ImageDraw.Draw(img)
current_page = "draw"   # "draw" hoặc "upload"
draw_canvas = None
img_canvas = None
result_draw_label = None
result_upload_label = None
upload_tk_img = None


# ================== CÁC HÀM CHUNG ==================
def paint(event):
    """Vẽ nét đen lên khung vẽ tay"""
    x, y = event.x, event.y
    draw_canvas.create_oval(x-PEN, y-PEN, x+PEN, y+PEN,
                            fill="black", outline="black")
    draw.ellipse((x-PEN, y-PEN, x+PEN, y+PEN), fill=0)


def clear_draw():
    """Xóa khung vẽ tay"""
    global img, draw
    img = Image.new("L", (CANVAS_W, CANVAS_H), 255)
    draw = ImageDraw.Draw(img)
    draw_canvas.delete("all")
    result_draw_label.config(text="Chưa có kết quả")

# Upload ảnh
def load_img():
    """Chọn ảnh cho TRANG ẢNH"""
    global img, draw, upload_tk_img
    path = filedialog.askopenfilename(
        filetypes=[("Ảnh", "*.png;*.jpg;*.jpeg;*.bmp")]
    )
    if not path:
        return

    loaded = Image.open(path).convert("L").resize((CANVAS_W, CANVAS_H))
    img = loaded               # cho predict() dùng chung
    draw = ImageDraw.Draw(img) # không bắt buộc nhưng giữ cấu trúc

    upload_tk_img = ImageTk.PhotoImage(loaded)
    img_canvas.delete("all")
    img_canvas.create_image(0, 0, anchor="nw", image=upload_tk_img)
    result_upload_label.config(text="Chưa có kết quả")


def predict():
    """Nhận dạng: dùng preprocess + model, hiển thị top 3"""
    if img is None:
        return

    arr = preprocess(img)
    x = np.expand_dims(arr, 0)
    preds = model.predict(x, verbose=0)[0]

    top = preds.argsort()[::-1][:3]
    txt = f"Kết quả: {top[0]} ({preds[top[0]]*100:.2f}%)\n"
    txt += f"{top[1]} ({preds[top[1]]*100:.2f}%)\n"
    txt += f"{top[2]} ({preds[top[2]]*100:.2f}%)"

    # Đang ở trang nào thì cập nhật label của trang đó
    if current_page == "draw":
        result_draw_label.config(text=txt)
    else:
        result_upload_label.config(text=txt)


def show_page(name):
    """Chuyển giữa 2 trang"""
    global current_page
    current_page = name
    if name == "draw":
        page_draw.tkraise()
    else:
        page_upload.tkraise()


# ======================= MAIN WINDOW =======================
root = tk.Tk()
root.title("NHẬN DẠNG CHỮ SỐ VIẾT TAY BẰNG CNN")
root.geometry("1100x600")
root.configure(bg="#a6ddf0")

# ====== TIÊU ĐỀ ======
title = tk.Label(root, text="CHƯƠNG TRÌNH NHẬN DẠNG CHỮ SỐ VIẾT TAY BẰNG CNN", font=("Arial", 22, "bold"), bg="#ffeb3b")
title.pack(fill="x", pady=5)

# ====== KHUNG CHỨA 2 TRANG ======
container = tk.Frame(root, bg="#a6ddf0")
container.pack(fill="both", expand=True, pady=10)

page_draw = tk.Frame(container, bg="#a6ddf0")
page_upload = tk.Frame(container, bg="#a6ddf0")

for p in (page_draw, page_upload):
    p.place(relwidth=1, relheight=1)

# ================== TRANG 1: VẼ TAY (HÌNH 1) ==================
page_draw.columnconfigure((0, 1), weight=1)

lbl_draw_left = tk.Label(page_draw, text="KHUNG VẼ", font=("Arial", 20, "bold"), bg="#a6ddf0")
lbl_draw_left.grid(row=0, column=0, pady=10)

lbl_draw_right = tk.Label(page_draw, text="KẾT QUẢ", font=("Arial", 20, "bold"),bg="#a6ddf0")
lbl_draw_right.grid(row=0, column=1, pady=10)

# Khung vẽ
frame_draw_canvas = tk.Frame(page_draw, bg="black", bd=3, relief="solid")
frame_draw_canvas.grid(row=1, column=0, padx=60, pady=10)
draw_canvas = tk.Canvas(frame_draw_canvas, width=CANVAS_W,height=CANVAS_H, bg="white")
draw_canvas.pack()
draw_canvas.bind("<B1-Motion>", paint)

# Khung kết quả
frame_draw_result = tk.Frame(page_draw, bg="white", bd=3, relief="solid", width=310, height=310)
frame_draw_result.grid(row=1, column=1, padx=60, pady=10)
frame_draw_result.pack_propagate(False)
result_draw_label = tk.Label(frame_draw_result, text="Chưa có kết quả", font=("Arial", 16), bg="white", justify="left")
result_draw_label.pack(expand=True)

# Hàng nút phía dưới
btn_frame_draw = tk.Frame(page_draw, bg="#a6ddf0")
btn_frame_draw.grid(row=2, column=0, columnspan=2, pady=25)


tk.Button(btn_frame_draw, text="NHẬN DẠNG", font=("Arial", 14), width=18, command=predict).pack(side="left", padx=15)
tk.Button(btn_frame_draw, text="XÓA", font=("Arial", 14), width=18, command=clear_draw).pack(side="left", padx=15)
tk.Button(btn_frame_draw, text="CHUYỂN SANG ẢNH", font=("Arial", 14), width=18, command=lambda: show_page("upload")).pack(side="left", padx=15)

# Khởi tạo khung vẽ trắng
clear_draw()


# ================== TRANG 2: ẢNH (HÌNH 2) ==================
page_upload.columnconfigure((0, 1), weight=1)

lbl_up_left = tk.Label(page_upload, text="ẢNH", font=("Arial", 20, "bold"), bg="#a6ddf0")
lbl_up_left.grid(row=0, column=0, pady=10)

lbl_up_right = tk.Label(page_upload, text="KẾT QUẢ", font=("Arial", 20, "bold"), bg="#a6ddf0")
lbl_up_right.grid(row=0, column=1, pady=10)

# Khung hiển thị ảnh
frame_img_canvas = tk.Frame(page_upload, bg="black", bd=3, relief="solid")
frame_img_canvas.grid(row=1, column=0, padx=60, pady=10)
img_canvas = tk.Canvas(frame_img_canvas, width=CANVAS_W, height=CANVAS_H, bg="white")
img_canvas.pack()

# Khung kết quả
frame_up_result = tk.Frame(page_upload, bg="white", bd=3, relief="solid", width=310, height=310)
frame_up_result.grid(row=1, column=1, padx=60, pady=10)
frame_up_result.pack_propagate(False)  # giữ khung to cố định

result_upload_label = tk.Label(frame_up_result, text="Chưa có kết quả", font=("Arial", 16), bg="white", justify="left")
result_upload_label.pack(expand=True)

# Hàng nút phía dưới
btn_frame_up = tk.Frame(page_upload, bg="#a6ddf0")
btn_frame_up.grid(row=2, column=0, columnspan=2, pady=25)

tk.Button(btn_frame_up, text="NHẬN DẠNG", font=("Arial", 14), width=18, command=predict).pack(side="left", padx=20)
tk.Button(btn_frame_up, text="UPLOAD ẢNH", font=("Arial", 14), width=18, command=load_img).pack(side="left", padx=20)
tk.Button(btn_frame_up, text="CHUYỂN SANG VẼ", font=("Arial", 14), width=18, command=lambda: show_page("draw")).pack(side="left", padx=20)

# Mặc định mở TRANG VẼ TAY
show_page("draw")

root.mainloop()
