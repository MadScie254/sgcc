import { useEffect, useState } from "react";
import { keepPreviousData, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { getModelMetrics, getOperatingCurve, publishThreshold, type OperatingPoint } from "@/lib/api";
import { useDebounced } from "@/lib/debounce";
import { fmtInt, fmtNum, fmtPct } from "@/lib/format";
import { useIsSupervisor } from "@/lib/me";
import { ApiError, Button, Card, CardHeader, PageHeader, Pill, Skeleton, Stat } from "@/components/ui";
import { ConfusionGrid, Legend, NetValueChart, TradeoffChart } from "@/components/charts";

function nearest(points: OperatingPoint[], threshold: number): number {
  return points.reduce((best, p, i) => (Math.abs(p.threshold - threshold) < Math.abs(points[best].threshold - threshold) ? i : best), 0);
}

function NumberField({ id, label, value, onChange, hint }: { id: string; label: string; value: string; onChange: (v: string) => void; hint: string }) {
  return (
    <div className="flex min-w-[150px] flex-1 flex-col gap-1">
      <label htmlFor={id} className="text-[13px] text-ink-2">{label}</label>
      <input id={id} type="number" inputMode="decimal" min={0} value={value} onChange={(e) => onChange(e.target.value)} placeholder="—"
        className="h-11 rounded-lg border border-line bg-surface px-3 font-mono text-sm tabular outline-none focus:border-cobalt" />
      <span className="text-xs text-ink-3">{hint}</span>
    </div>
  );
}

const positive = (text: string): number | undefined => {
  const value = Number(text);
  return text.trim() && Number.isFinite(value) && value > 0 ? value : undefined;
};

export function ThresholdStudioPage() {
  const queryClient = useQueryClient();
  const supervisor = useIsSupervisor();
  const [capacity, setCapacity] = useState("");
  const [cost, setCost] = useState("");
  const [value, setValue] = useState("");
  const costModel = useDebounced({ capacity: positive(capacity), cost_per_visit: positive(cost), value_per_theft: positive(value) });
  const curve = useQuery({
    queryKey: ["operating-curve", costModel],
    queryFn: () => getOperatingCurve({ ...costModel, capacity: costModel.capacity && Math.round(costModel.capacity) }),
    placeholderData: keepPreviousData,
  });
  const metrics = useQuery({ queryKey: ["model-metrics"], queryFn: getModelMetrics });
  const [index, setIndex] = useState<number | null>(null);

  useEffect(() => {
    if (index === null && curve.data && metrics.data) setIndex(nearest(curve.data.points, metrics.data.threshold));
  }, [index, curve.data, metrics.data]);

  const publish = useMutation({
    mutationFn: publishThreshold,
    onSuccess: () => queryClient.invalidateQueries(),
  });

  if (curve.isError || metrics.isError) return <ApiError error={curve.error ?? metrics.error} />;
  if (!curve.data || !metrics.data || index === null) {
    return <div className="flex flex-col gap-4"><Skeleton className="h-24" /><Skeleton className="h-28" /><Skeleton className="h-80" /></div>;
  }

  const { points } = curve.data;
  const p = points[index];
  const inService = metrics.data.threshold;
  const trained = metrics.data.trained_threshold;
  const isInService = Math.abs(p.threshold - inService) < 0.005;
  const hasValue = Boolean(costModel.cost_per_visit || costModel.value_per_theft);
  // Among thresholds with the same net value (e.g. once the capacity is filled), the highest flags fewest customers.
  const best = points.reduce((b, pt, i) => (pt.net_value >= points[b].net_value - 1e-9 ? i : b), 0);
  const presets: Array<[string, number]> = [["Wide net", 0.1], ["Trained", trained], ["More likely than not", 0.5]];
  if (hasValue) presets.push(["Best net value", points[best].threshold]);

  return (
    <>
      <PageHeader
        eyebrow={`Validation customers: ${fmtInt(curve.data.validation_customers)} (${fmtInt(curve.data.validation_theft)} thieves) · population: ${fmtInt(curve.data.population_customers)} customers`}
        title="Threshold studio"
        description="Choose how many customers to inspect. Hit rate and recall come from the validation customers the threshold is chosen on; workload and expected thefts are for the population in service. The test customers are never used here."
      />

      <Card label="Threshold control" className="flex flex-col gap-5 px-6 py-5 lg:flex-row lg:items-center lg:gap-7">
        <div className="flex w-[150px] shrink-0 flex-col gap-0.5">
          <span className="text-[13px] text-ink-2">Threshold τ</span>
          <span className="font-display text-[44px] font-medium leading-tight tabular">{p.threshold.toFixed(2)}</span>
          <span className="text-xs text-ink-3">in service {inService.toFixed(3)}</span>
        </div>
        <div className="flex flex-1 flex-col gap-2">
          <label htmlFor="threshold-range" className="sr-only">Decision threshold</label>
          <input id="threshold-range" type="range" min={0} max={points.length - 1} step={1} value={index}
            onChange={(e) => setIndex(Number(e.target.value))} className="studio-range h-7 w-full cursor-pointer"
            aria-valuetext={`threshold ${p.threshold.toFixed(2)}`} />
          <span className="flex justify-between text-xs text-ink-3"><span>Catch more theft</span><span>Fewer wasted visits</span></span>
        </div>
        <div role="group" aria-label="Presets" className="flex flex-wrap gap-2">
          {presets.map(([label, threshold]) => {
            const i = nearest(points, threshold);
            return <Pill key={label} active={i === index} onClick={() => setIndex(i)}>{label} · {points[i].threshold.toFixed(2)}</Pill>;
          })}
        </div>
      </Card>

      <Card label="Cost model" className="flex flex-col gap-4 px-6 py-5">
        <CardHeader title="Inspection budget" subtitle="Optional. With a capacity, the highest-ranked customers are visited first." />
        <div className="flex flex-wrap gap-4">
          <NumberField id="capacity" label="Inspections the team can make" value={capacity} onChange={setCapacity} hint="blank: visit every flagged customer" />
          <NumberField id="cost" label="Cost per visit" value={cost} onChange={setCost} hint="truck roll, staff time" />
          <NumberField id="value" label="Value recovered per theft" value={value} onChange={setValue} hint="back-billing, avoided loss" />
        </div>
      </Card>

      <section aria-label="Outcome figures" className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <Stat label="Visits" value={fmtInt(p.visits)} hint={p.visits < p.population_flagged ? `of ${fmtInt(p.population_flagged)} flagged (capacity)` : "every flagged customer"} />
        <Stat label="Expected thefts found" tone="risk" value={fmtNum(p.expected_thefts_found, 0)} hint="estimate from calibrated probabilities" />
        <Stat label="Hit rate (validation)" value={fmtPct(p.precision)} hint={`recall ${fmtPct(p.recall)} of validation thieves`} />
        <Stat label="Net value" value={hasValue ? fmtInt(Math.round(p.net_value)) : "—"} hint={hasValue ? "expected recovered value minus visit cost" : "enter costs above"} />
      </section>

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-[1.3fr_1fr]">
        <Card label="Precision and recall" className="flex flex-col gap-3 p-[22px]">
          <CardHeader title="Trade-off on validation customers" subtitle="Click the chart to jump to a threshold."
            action={<Legend items={[{ color: "#2346C8", label: "Hit rate (precision)", line: true }, { color: "#E0873A", label: "Thefts caught (recall)", line: true }]} />} />
          <TradeoffChart points={points} index={index} onPick={setIndex} trainedThreshold={trained} />
          {hasValue ? (
            <>
              <CardHeader title="Expected net value in the population" />
              <NetValueChart points={points} index={index} />
            </>
          ) : null}
        </Card>

        <Card label="Confusion matrix" className="flex flex-col gap-4 p-[22px]">
          <CardHeader title="Validation customers at this threshold" subtitle="What these settings did on customers with known outcomes" />
          <ConfusionGrid tp={p.tp} fp={p.fp} fn={p.fn} tn={p.tn} />
          <div className="mt-auto flex flex-col gap-2.5">
            {publish.isError ? <ApiError title="Could not publish" error={publish.error} /> : null}
            {publish.isSuccess && isInService ? (
              <p role="status" className="m-0 text-[13px] text-cobalt-ink">Published. Scoring now flags {fmtInt(p.population_flagged)} customers; new flags open cases.</p>
            ) : null}
            <Button variant="primary" disabled={isInService || !supervisor} busy={publish.isPending} onClick={() => publish.mutate(p.threshold)}>
              {isInService ? `τ ${p.threshold.toFixed(2)} is in service` : `Publish τ ${p.threshold.toFixed(2)} to scoring`}
            </Button>
            {!supervisor ? <p className="m-0 text-xs text-ink-3">Publishing a threshold needs the supervisor role.</p> : null}
            {supervisor && Math.abs(inService - trained) > 0.005 ? (
              <Button variant="ghost" busy={publish.isPending} onClick={() => publish.mutate(null)}>Return to trained threshold ({trained.toFixed(3)})</Button>
            ) : null}
          </div>
        </Card>
      </div>
    </>
  );
}
