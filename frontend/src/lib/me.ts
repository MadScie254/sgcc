import { useQuery } from "@tanstack/react-query";
import { getMe } from "@/lib/api";

/** Who the API key belongs to, and their role; drives which actions the console offers. */
export function useMe() {
  return useQuery({ queryKey: ["me"], queryFn: getMe, staleTime: 60_000, retry: false });
}

export function useIsSupervisor(): boolean {
  return useMe().data?.role === "supervisor";
}
