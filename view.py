from tkinter import *
from tkinter import filedialog
from tkinter import messagebox
import pandas as pd
from model import model
import os

"""
This class responsible for the GUI.
Including all the objects on the screen and their event handlers.
"""
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
        """
        Validates that the user inserted number of bins.
        :param new_text:
        :return:
        """
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
        """
        Handler for browse button.
        Extracts the given directory path.
        :return:
        """
        try:
            self.path_to_directory = filedialog.askdirectory()
            self.directory_path.delete('1.0', END)
            self.directory_path.insert(END, self.path_to_directory)
            self.path_flag = True
        except:
            pass


    def clik_on_build(self):
        """
        Handler for build button.
        If all the requiered parameters were inserted well, the function will define a classifier and build a model.
        :return:
        """
        self.enable_claasify()
        missing = self.model.validate_files(os.listdir(self.path_to_directory))
        if len(missing) > 0:
             messagebox.showerror("Naive Bayes Classifier", "you must have this files " + str(missing))
             return
        with open(self.path_to_directory + '\Structure.txt', "r") as f:

            try:
                train = pd.read_csv(self.path_to_directory + "\\train.csv")
            except:
                messagebox.showerror("Naive Bayes Classifier", "the file train.csv has no data!")
                return
            file_content = f.readlines()
            if len(file_content) > 0:
                self.model.create_classifier(file_content, train, self.entered_number)
            else:
                messagebox.showerror("Naive Bayes Classifier", "the file Structure.txt has no data!")

            messagebox.showinfo("Naive Bayes Classifier", "Building classifier using train-set is done!")

    def clik_on_classify(self):
        """
        Handler for classify button.
        Creates new output file, and run the classification process.
        In the end of the process the classification results will be on the output file.
        :return:
        """
        try:
            test = pd.read_csv(self.path_to_directory + "\\test.csv")
        except:
            messagebox.showerror("Naive Bayes Classifier", "the file test.csv has no data!")
            return
        output_path =self.path_to_directory + '\output.txt'
        self.model.execute_classification(test,output_path)
        messagebox.showinfo("Naive Bayes Classifier", "Classification finish successfully! the output file is in "
                            +self.path_to_directory+"/output.txt")
        exit(0)

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