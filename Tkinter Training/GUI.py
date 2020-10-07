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


frm_form = tk.Frame(relief=tk.SUNKEN, bg="black")
frm_form.pack(fill=tk.BOTH)
frm_form.columnconfigure(1, weight=1, minsize=250)


# window.rowconfigure(, minsize=100)

label1 = tk.Label(master=frm_form, text="First Name:")
label2 = tk.Label(master=frm_form, text="Address 1:")
label3 = tk.Label(master=frm_form, text="Address 2:")
label4 = tk.Label(master=frm_form, text="City:")
label5 = tk.Label(master=frm_form, text="State/Province:")
label6 = tk.Label(master=frm_form, text="Postal Code:")
label7 = tk.Label(master=frm_form, text="Last Name:")
label8 = tk.Label(master=frm_form, text="Country:")

rat = [label1, label2, label3, label4, label5, label6, label7, label8]

for j in range(7):
    rat[j].grid(row=j, column=0, sticky="n")


for i in range(7):
    ent = tk.Entry(master=frm_form)
    ent.grid(row=i, column=1, sticky="we")


frm_footer = tk.Frame(relief=tk.RAISED)
frm_footer.pack()

btn_clear = tk.Button(master=frm_footer, text="Clear")
btn_clear.grid()

window.mainloop()

# ex Write a complete script that displays an Entry widget that’s 40 text units wide and has a white background and black text. Use .insert() to display text in the widget that reads "What is your name?".
# entry = tk.Entry(width=40, bg="white", fg="black")
# entry.pack()

# entry.insert(0, "What is your name?")
