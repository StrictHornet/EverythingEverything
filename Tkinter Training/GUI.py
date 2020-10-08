import tkinter as tk

# GUI
# win = tk.Tk()

# frm_a = tk.Frame()
# frm_a.pack()

# lbl_greeting = tk.Label(text="Hello there!")
# lbl_greeting.pack()

# lbl_inq_approval = tk.Label(text="Inq Approval Alphabet Index",
#                             master=frm_a, relief=tk.SUNKEN)
# lbl_inq_approval.pack()

# ent1 = tk.Entry()
# ir21 = ent1.get()

# ent2 = tk.Entry()
# ir22 = ent2.get()

# ent1.pack()
# ent2.pack()
# # print(approval_index)

# rec = tk.Button(text="Reconcile", bg="red", width=15, height=3)
# rec.pack()

# for i in range(3):
#     win.columnconfigure(i, weight=1, minsize=75)
#     win.rowconfigure(i, weight=1, minsize=50)

#     for j in range(0, 3):
#         frame = tk.Frame(
#             master=win,
#             relief=tk.RAISED,
#             borderwidth=1
#         )
#         frame.grid(row=i, column=j, padx=5, pady=5)

#         label = tk.Label(master=frame, text=f"Row {i}\nColumn {j}")
#         label.pack(padx=5, pady=5)

window = tk.Tk()


frm_form = tk.Frame(relief=tk.SUNKEN)
frm_form.pack(fill=tk.BOTH)
frm_form.columnconfigure(1, weight=1, minsize=250)

label = [
    "First Name:",
    "Last Name:",
    "Address 1:",
    "Address 2:",
    "City:",
    "State/Province:",
    "Postal Code:",
    "Country:"
]

for i, label_text in enumerate(label):
    label_loop = tk.Label(master=frm_form, text=label_text)
    label_loop.grid(row=i, column=0)

    ent = tk.Entry(master=frm_form)
    ent.grid(row=i, column=1, sticky="we")

frm_footer = tk.Frame(relief=tk.RAISED,  bg="black")
frm_footer.pack(fill=tk.X, ipadx=5, ipady=5)

btn_clear = tk.Button(master=frm_footer, text="Clear")
btn_clear.pack(side=tk.RIGHT)

window.mainloop()

# ex Write a complete script that displays an Entry widget that’s 40 text units wide and has a white background and black text. Use .insert() to display text in the widget that reads "What is your name?".
# entry = tk.Entry(width=40, bg="white", fg="black")
# entry.pack()

# entry.insert(0, "What is your name?")
