from tkinter import *
from tkinter import filedialog
from tkinter import messagebox
import pandas as pd
from model import model

class view(object):

    def __init__(self, master, model):
        #view classss members
        self.model = model
        self.master = master
        self.path_to_directory= ""
        self.cuurent_model="none"
        self.bin_num = master.register(self.validate) # we have to wrap the command
        self.path_flag = False
        self.bin_flag = False
        #widdgets on the gui
        self.directory_path = Text(self.master, height=2, width=50)
        # self.bins_input = Text(self.master, height=2, width=50)
        self.bins_input = Entry(master, validate="key", validatecommand=(self.bin_num, '%P'))
        self.b_browse_directory=Button(self.master, compound=LEFT, text="Browse", command=lambda : self.clik_on_browse_directory())
        self.b_build=Button(self.master, compound=CENTER, text="Build", command=lambda : self.clik_on_build())
        self.b_classify=Button(self.master, compound=CENTER, text="Classify", command=lambda : self.clik_on_classify())

        #widget placment
        Label(self.master, text="Directory path: ", compound=LEFT).grid(row=0, column=0,sticky=W)
        self.directory_path.grid(row=0, column=1, sticky=W)
        self.b_browse_directory.grid(row=0, column=2, sticky=N + S + E + W)
        Label(self.master, text="Discretization Bins", compound=LEFT).grid(row=1, column=0,sticky=W)
        self.bins_input.grid(row=1, column=1,sticky=W,pady=15)
        self.b_build.grid(row=2, column=1, pady=15)
        self.b_classify.grid(row=3, column=1, pady=15)
        self.b_build.config(state=DISABLED)
        self.b_classify.config(state=DISABLED)

    def validate(self, new_text):
        ans = False
        if not new_text:  # the field is being cleared
            self.entered_number = 0
            self.bin_flag = True
            ans = True

        try:
            self.entered_number = int(new_text)
            self.bin_flag = True
            ans = True
        except ValueError:
            self.bin_flag = False
            ans = False
        if self.bin_flag and self.path_flag:
            self.enable_build()
        return ans


    def clik_on_browse_directory(self):
        try:
            self.path_to_directory = filedialog.askdirectory()
            self.directory_path.delete('1.0', END)
            self.directory_path.insert(END, self.path_to_directory)
            self.path_flag = True
        except:
            pass


    def clik_on_build(self):
        self.enable_claasify()
        with open(self.path_to_directory + '\Structure.txt', "r") as f:
            test = pd.read_csv(self.path_to_directory+"\\train.csv")
            self.model.create_classifier(f.readlines(),test,self.entered_number)

    def clik_on_classify(self):
        pass

    def enable_build(self):
        self.b_build.config(state=NORMAL)

    def enable_claasify(self):
        self.b_classify.config(state=NORMAL)


def main():
    root = Tk()
    root.title("Naive Bayes Classifier")
    root.geometry("600x200")
    root.resizable(0,0)
    Model = model()
    View = view(root, Model)
    root.mainloop()



if __name__ == '__main__':
    main()