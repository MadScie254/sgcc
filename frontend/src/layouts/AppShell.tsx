import type { FormEvent, ReactNode } from "react";
import { useMemo, useState } from "react";
import { useNavigate, NavLink } from "react-router-dom";
import { Activity, BarChart3, LayoutDashboard, Search, Settings, Target, Users } from "lucide-react";
import { useQuery } from "@tanstack/react-query";
import { getHealth } from "@/lib/api";
import { cn } from "@/components/ui/primitives";

const navItems = [
  { to: "/", label: "Overview", icon: LayoutDashboard },
  { to: "/predict", label: "Predict", icon: Target },
  { to: "/explain", label: "Explain", icon: BarChart3 },
  { to: "/monitor", label: "Monitor", icon: Activity },
  { to: "/customers", label: "Customers", icon: Users },
];

export function AppShell({ children }: { children: ReactNode }) {
  const navigate = useNavigate();
  const [search, setSearch] = useState("");
  const healthQuery = useQuery({ queryKey: ["health"], queryFn: getHealth, refetchInterval: 30_000 });

  const modelLive = healthQuery.isSuccess && healthQuery.data?.status === "ok";
  const statusText = modelLive ? "Model live" : healthQuery.isError ? "API offline" : "Checking...";

  const avatar = useMemo(() => "SG", []);

  function submitSearch(event: FormEvent) {
    event.preventDefault();
    const value = search.trim();
    if (!value) return;
    navigate(`/customers?search=${encodeURIComponent(value)}`);
  }

  return (
    <div className="min-h-screen bg-bg text-primary">
      <div className="grid min-h-screen lg:grid-cols-[260px_1fr]">
        <aside className="border-b border-border bg-surface lg:min-h-screen lg:border-b-0 lg:border-r">
          <div className="flex h-full flex-col px-4 py-5 lg:px-5">
            <div className="space-y-1 border-b border-border pb-4">
              <div className="text-xs uppercase tracking-[0.22em] text-secondary">SGCC Theft Detector</div>
              <div className="text-xl font-semibold tracking-tight text-primary">Operations Console</div>
              <div className="text-sm leading-6 text-secondary">A compact dashboard for risk review, prediction, explainability, and monitoring.</div>
            </div>

            <nav className="mt-4 flex flex-1 flex-col gap-1">
              {navItems.map((item) => {
                const Icon = item.icon;
                return (
                  <NavLink
                    key={item.to}
                    to={item.to}
                    end={item.to === "/"}
                    className={({ isActive }) =>
                      cn(
                        "flex items-center gap-3 rounded-md border px-3 py-3 text-sm font-medium transition",
                        isActive
                          ? "border-accent bg-accent-bg text-accent"
                          : "border-transparent text-secondary hover:border-border hover:bg-surface-alt hover:text-primary",
                      )
                    }
                  >
                    <Icon className="h-4 w-4" />
                    <span>{item.label}</span>
                  </NavLink>
                );
              })}

              <div className="mt-auto border-t border-border pt-2">
                <NavLink
                  to="/settings"
                  className={({ isActive }) =>
                    cn(
                      "flex items-center gap-3 rounded-md border px-3 py-3 text-sm font-medium transition",
                      isActive
                        ? "border-accent bg-accent-bg text-accent"
                        : "border-transparent text-secondary hover:border-border hover:bg-surface-alt hover:text-primary",
                    )
                  }
                >
                  <Settings className="h-4 w-4" />
                  <span>Settings</span>
                </NavLink>
              </div>
            </nav>
          </div>
        </aside>

        <main className="flex min-w-0 flex-col">
          <header className="border-b border-border bg-surface px-4 py-4 lg:px-6">
            <div className="flex flex-col gap-4 xl:flex-row xl:items-center xl:justify-between">
              <form onSubmit={submitSearch} className="flex max-w-xl flex-1 items-center gap-3 rounded-md border border-border bg-surface px-3 py-2">
                <Search className="h-4 w-4 text-muted" />
                <input
                  value={search}
                  onChange={(event) => setSearch(event.target.value)}
                  placeholder="Search customer ID"
                  className="w-full border-0 bg-transparent text-sm text-primary outline-none placeholder:text-muted"
                />
              </form>

              <div className="flex items-center gap-3 self-end xl:self-auto">
                <span className={cn("inline-flex items-center gap-2 rounded-full border px-3 py-1 text-xs font-semibold", modelLive ? "border-success bg-success-bg text-success" : "border-danger bg-danger-bg text-danger") }>
                  <span className={cn("h-2 w-2 rounded-full", modelLive ? "bg-success" : "bg-danger")} />
                  {statusText}
                </span>
                <div className="flex h-10 w-10 items-center justify-center rounded-full border border-border bg-accent-bg text-sm font-semibold text-accent">
                  {avatar}
                </div>
              </div>
            </div>
          </header>

          <section className="min-w-0 flex-1 px-4 py-5 lg:px-6 lg:py-6">
            {children}
          </section>
        </main>
      </div>
    </div>
  );
}
