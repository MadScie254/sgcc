import { create } from "zustand";

type AppState = {
  selectedCustomerId: string;
  threshold: number;
  modelVersion: string;
  setSelectedCustomerId: (customerId: string) => void;
  setThreshold: (threshold: number) => void;
  setModelVersion: (modelVersion: string) => void;
};

export const useAppStore = create<AppState>((set) => ({
  selectedCustomerId: "",
  threshold: 0.5,
  modelVersion: "xgb_best",
  setSelectedCustomerId: (selectedCustomerId) => set({ selectedCustomerId }),
  setThreshold: (threshold) => set({ threshold }),
  setModelVersion: (modelVersion) => set({ modelVersion }),
}));