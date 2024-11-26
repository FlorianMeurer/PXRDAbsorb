import tkinter as tk
from tkinter import ttk, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import brennan as br
from sklearn.preprocessing import normalize
import json
import scipy as sp
from scipy import constants as conts
import os
import csv
import re

def get_atomic_absorption_coefficient(element, energy):
    bren = br.brennan()
    mu = bren.get_mu_at_angstrom(ret_wl(energy),element) 
    return mu

def ret_energy(wavelength):
    return (conts.h * conts.c) / (wavelength*1.60218e-19) * 1E10

def ret_wl(energy):
    return (conts.h * conts.c) / (energy*1.60218e-19) * 1E10

def load_atomic_masses(file_path):
    with open(file_path, 'r') as file:
        atomic_masses = json.load(file)
    return atomic_masses

def calculate_compound_mu(formula, energy, atomic_masses, density, Z=4) :   
    elements = re.findall(r'([A-Z][a-z]*)(\d*\.?\d*)', formula)
    molecular_weight = 0
    for (element, count) in elements:
        count = count if count else 1
        molecular_weight += atomic_masses[element] * float(count)
    
    molecular_weight_cell = molecular_weight*Z
    mu_per_AU = 0
        
    for (element, count) in elements:
        count = float(count) if count else 1
        mu_at = get_atomic_absorption_coefficient(element, energy)/1e24
        mu_per_AU += mu_at * count
    
    mu_per_cell = mu_per_AU*Z
    mu = density * mu_per_cell / molecular_weight_cell * 6.022e23 /10
    return mu

def parse_formula(formula):
    elements = re.findall(r'([A-Z][a-z]*)(\d*\.?\d*)', formula)
    composition = {}
    for element, count in elements:
        if count == '':
            count = int(1)
        else:
            count = float(count)
        composition[element] = count
    return composition

def molecular_weight(formula, atomic_masses):
    composition = parse_formula(formula)
    weight = 0
    for elem in composition:
        weight += atomic_masses[elem] * composition[elem]
    #weight = sum(atomic_masses[element] * count for element, count in composition.items())
    return weight

def plot_mu_vs_energy(formula,wavelength,rho,z1,grid = 100):
    import re
    bren = br.brennan()
    atomic_masses = load_atomic_masses("elemental_masses.json")
    elements = re.findall(r'([A-Z][a-z]*)(\d*)', formula)
    if wavelength > 2:
        wavelength = ret_wl(wavelength)
    if ret_energy(wavelength) < 30000:
        energies = np.arange(10000,30000,grid)
    elif ret_energy(wavelength) < 10000:
        energies = np.arange(2000,6000,grid)
    else:
        energies = np.arange(30000,70000,grid)
    
    elem_dict = {}
    for elem in elements:
        fdps = [] 
        for ener in energies:
            wl = ret_wl(ener)
            _, fdp = bren.at_angstrom(wl,elem[0])
            fdps.append(fdp)
        elem_dict[elem[0]] = fdps
    
    mus = []
    for ener in energies:
        mu = calculate_compound_mu(formula,ener,atomic_masses,rho,z1)
        mus.append(mu)
    return energies, mus, elem_dict
    
        
def optimize_scattering_intensity(rho1, rho2, formula1, formula2, z1, z2, energy, thickness_in_mm,packing, df=None, fraction_type='mass'):
    x = thickness_in_mm    
    atomic_masses = load_atomic_masses("elemental_masses.json")
    atomic_numbers = load_atomic_masses("atomic_numbers.json")
    elements = re.findall(r'([A-Z][a-z]*)(\d*)', formula1)
    P1 = atomic_numbers[f"{elements[0][0]}"]
    
    mu1 = calculate_compound_mu(formula1,energy,atomic_masses,rho1,z1)
    mu2 = calculate_compound_mu(formula2,energy,atomic_masses,rho2,z2)
    mw1 = molecular_weight(formula1, atomic_masses)
    mw2 = molecular_weight(formula2, atomic_masses)
    
    print(f"Found mu = {mu1:.3} $mm^{-1}$ for {formula1}\nFound mu = {mu2:.3} $mm^{-1}$ for {formula2}")

    if fraction_type == 'volume':
        frac1_array = np.linspace(0, 1, 100)
        frac2_array = 1 - frac1_array
    elif fraction_type == 'mass':
        frac1_array = np.linspace(0, 1, 100)
        frac2_array = 1 - frac1_array
        frac1_array = frac1_array / (frac1_array + frac2_array * (rho1 / rho2))
        frac2_array = 1 - frac1_array
    elif fraction_type == 'molar':
        frac1_array = np.linspace(0, 1, 100)
        frac2_array = 1 - frac1_array
        frac1_array = frac1_array / (frac1_array + frac2_array * (mw1 / mw2))
        frac2_array = 1 - frac1_array

    I_array = []
    T_array = []
    P_array = []

    for frac1, frac2 in zip(frac1_array, frac2_array):
        if fraction_type == 'volume':
            mu_eff = frac1 * mu1 + frac2 * mu2
            P_total = frac1 * P1
        elif fraction_type == 'mass':
            mu_eff = frac1 * mu1 + frac2 * mu2
            P_total = frac1 * P1
        elif fraction_type == 'molar':
            mol1 = frac1 / mw1
            mol2 = frac2 / mw2
            mol_total = mol1 + mol2
            frac1_mol = mol1 / mol_total
            frac2_mol = mol2 / mol_total
            mu_eff = frac1_mol * mu1 + frac2_mol * mu2
            P_total = frac1_mol * P1
        T = np.exp(-mu_eff * x * packing)
        I = P_total * T
        I_array.append(I)
        T_array.append(T)
        P_array.append(P_total)

    I_array = np.array(I_array)
    optimal_index = np.argmax(I_array)
    optimal_frac1 = frac1_array[optimal_index]
    optimal_intensity = I_array[optimal_index]

    if df is not None:
        vols = df.VolThO2
        normalized_df = (df["3"] - df["3"].min()) / (df["3"].max() - df["3"].min()) * max(I_array)
        plt.plot(vols / 100, normalized_df, "x", color="black", label="Exp. ThO2 - 300um")

    return frac1_array, I_array, T_array, P_array, optimal_frac1, max(I_array)

  
class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("XRD Capilary Calc.")
        self.geometry("1200x800")
        self.config_file = "config.json"
        self.default_values = self.load_default_values()
        self.control_panel = ttk.Frame(self)
        self.control_panel.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        self.graph_panel = ttk.Frame(self)
        self.graph_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.create_control_panel()
        self.create_graph_panel()
        self.create_menu()
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def create_control_panel(self):
        labels = [
            ("Sum Formula 1", ""), ("Sum Formula 2", ""), 
            ("Density 1", "g/cm³"), ("Density 2", "g/cm³"), 
            ("Z Value 1", ""), ("Z Value 2", ""), 
            ("Capillary Thickness", "μm"), 
            ("Excitation Energy", "eV"), 
            ("Packing Parameter", "")
        ]

        self.entries = {}

        for i, (label, unit) in enumerate(labels):
            ttk.Label(self.control_panel, text=label).grid(row=i, column=0, padx=5, pady=5)
            entry = ttk.Entry(self.control_panel)
            entry.insert(0, self.default_values.get(label, ""))
            entry.grid(row=i, column=1, padx=5, pady=5)
            ttk.Label(self.control_panel, text=unit).grid(row=i, column=2, padx=5, pady=5)
            self.entries[label] = entry

        self.plot_button = ttk.Button(self.control_panel, text="Plot Graphs", command=self.plot_graphs)
        self.plot_button.grid(row=len(labels), column=0, columnspan=3, pady=10)

    def create_graph_panel(self):
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.graph_panel)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.graph_panel)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def create_menu(self):
        menubar = tk.Menu(self)
        self.config(menu=menubar)
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Save Graphs", command=self.save_graphs)
        file_menu.add_command(label="Save Data", command=self.save_data)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.on_closing)

    def plot_graphs(self):
        sumformula1 = self.entries["Sum Formula 1"].get()
        sumformula2 = self.entries["Sum Formula 2"].get()
        density1 = float(self.entries["Density 1"].get())
        density2 = float(self.entries["Density 2"].get())
        Z1 = float(self.entries["Z Value 1"].get())
        Z2 = float(self.entries["Z Value 2"].get())
        capillary_thickness = float(self.entries["Capillary Thickness"].get())
        excitation_energy = float(self.entries["Excitation Energy"].get())
        packing_parameter = float(self.entries["Packing Parameter"].get())

        energies, mus, elem_dict = plot_mu_vs_energy(sumformula1,excitation_energy,density1,Z1,grid=50)
        V1_array, I_array, T_array, P_array, optimal_V1, maxim = optimize_scattering_intensity(density1,density2,sumformula1,sumformula2,Z1,Z2,excitation_energy,capillary_thickness,packing_parameter)
        
        self.ax1.clear()
        for elem in elem_dict:
            self.ax1.plot(energies, elem_dict[elem], label=f"{elem}")
        self.ax1.plot(energies, mus, label= f"Total μ for {sumformula1}", color = "black", lw = 2)
        self.ax1.axvline(excitation_energy, color= "black", ls = "--", label =f"{excitation_energy:.2} eV")
        self.ax1.set_ylabel("μ / mm$^{-1}$")
        self.ax1.set_xlabel("Energy / eV")
        self.ax1.grid(alpha=0.5)
        self.ax1.legend()
        
        self.ax2.clear()
        self.ax2.plot(V1_array, I_array, label= "Effective Scattering Power")
        self.ax2.plot(V1_array, T_array, label = "Transmission")
        self.ax2.plot(V1_array, P_array, label = "Scattering Power")
        self.ax2.axvline(optimal_V1, color='grey', linestyle='--', label=f'Optimal V1 = {optimal_V1:.2f}', lw=1.2)
        self.ax2.set_ylim(-0.2,maxim*1.2)
        self.ax2.set_ylabel("Rel. Intensity")
        self.ax2.set_xlabel("Fraction")
        self.ax2.grid(alpha=0.5)
        self.ax2.legend()

        self.canvas.draw()
        self.geometry("1300x900")

    def save_graphs(self):
        files = [('PNG Image', '*.png'), ('All Files', '*.*')]
        file_path = filedialog.asksaveasfilename(filetypes=files, defaultextension=files)
        if file_path:
            self.fig.savefig(file_path)

    def save_data(self):
        files = [('CSV File', '*.csv'), ('All Files', '*.*')]
        file_path = filedialog.asksaveasfilename(filetypes=files, defaultextension=files)
        if file_path:
            with open(file_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Absorption X", "Absorption Y"])
                writer.writerows(zip(self.x1, self.y1))
                writer.writerow([])
                writer.writerow(["Scattering X", "Scattering Y"])
                writer.writerows(zip(self.x2, self.y2))

    def on_closing(self):
        self.save_default_values()
        self.quit()  
        self.destroy()  

    def load_default_values(self):
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as file:
                return json.load(file)
        else:
            return {
                "Sum Formula 1": "UO2",
                "Sum Formula 2": "SiO2",
                "Density 1": "10.87",
                "Density 2": "2.56",
                "Z Value 1": "4",
                "Z Value 2": "4",
                "Capillary Thickness": "0.4",
                "Excitation Energy": "20000",
                "Packing Parameter": "0.5"
            }

    def save_default_values(self):
        values = {label: entry.get() for label, entry in self.entries.items()}
        with open(self.config_file, 'w') as file:
            json.dump(values, file)

if __name__ == "__main__":
    app = Application()
    app.mainloop()