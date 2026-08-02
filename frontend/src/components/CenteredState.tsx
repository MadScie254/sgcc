import type { ReactNode } from "react";
import { AlertTriangle } from "lucide-react";

type CenteredStateProps = {
  title: string;
  description: string;
  icon?: ReactNode;
};

export function CenteredState({ title, description, icon }: CenteredStateProps) {
  return (
    <div className="flex min-h-[320px] items-center justify-center rounded-lg border border-border bg-surface px-6 py-10">
      <div className="max-w-md text-center">
        <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-full border border-border bg-surface-alt text-secondary">
          {icon ?? <AlertTriangle className="h-5 w-5" />}
        </div>
        <h3 className="mt-4 text-base font-semibold text-primary">{title}</h3>
        <p className="mt-2 text-sm leading-6 text-secondary">{description}</p>
      </div>
    </div>
  );
}
