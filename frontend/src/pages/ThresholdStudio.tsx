import { useEffect, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { apiErrorMessage, getModelMetrics, getOperatingCurve, publishThreshold, type OperatingPoint } from "@/lib/api";
import { fmtInt, fmtPct } from "@/lib/format";
import { Button, Card, CardHeader, ErrorState, PageHeader, Pill, Skeleton, Stat } from "@/components/ui";
import { ConfusionGrid, Legend, TradeoffChart } from "@/components/charts";

function nearest(points: OperatingPoint[], threshold: number): number {
  return points.reduce((best, p, i) => (Math.abs(p.threshold - threshold) < Math.abs(points[best].threshold - threshold) ? i : best), 0);
}

export function ThresholdStudioPage() {
  const queryClient = useQueryClient();
  const curve = useQuery({ queryKey: ["operating-curve"], queryFn: getOperatingCurve });
  const metrics = useQuery({ queryKey: ["model-metrics"], queryFn: getModelMetrics });
  const [index, setIndex] = useState<number | null>(null);

  useEffect(() => {
    if (index === null && curve.data && metrics.data) setIndex(nearest(curve.data, metrics.data.threshold));
  }, [index, curve.data, metrics.data]);

  const publish = useMutation({
    mutationFn: publishThreshold,
    onSuccess: () => queryClient.invalidateQueries(),
  });

  if (curve.isError || metrics.isError) return <ErrorState message={apiErrorMessage(curve.error ?? metrics.error)} />;
  if (!curve.data || !metrics.data || index === null) {
    return <div className="flex flex-col gap-4"><Skeleton className="h-24" /><Skeleton className="h-28" /><Skeleton className="h-80" /></div>;
  }

  const points = curve.data;
  const p = points[index];
  const theft = p.tp + p.fn;
  const inService = metrics.data.threshold;
  const trained = metrics.data.trained_threshold ?? inService;
  const isInService = Math.abs(p.threshold - inService) < 0.005;
  const presets: Array<[string, number]> = [["Wide net", 0.1], ["Balanced", trained], ["Sure bets", 0.5]];

  return (
    <>
      <PageHeader
        eyebrow={`${fmtInt(p.tp + p.fp + p.fn + p.tn)} held-out customers · ${fmtInt(theft)} known thefts`}
        title="Threshold studio"
        description="Every inspection costs a truck roll. Move the threshold to trade thefts caught against wasted visits, then publish it to the scoring pipeline."
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
          {presets.map(([label, value]) => {
            const i = nearest(points, value);
            return <Pill key={label} active={i === index} onClick={() => setIndex(i)}>{label} · {points[i].threshold.toFixed(2)}</Pill>;
          })}
        </div>
      </Card>

      <section aria-label="Outcome figures" className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <Stat label="Inspections" value={fmtInt(p.tp + p.fp)} hint="customers flagged" />
        <Stat label="Thefts caught" tone="risk" value={fmtInt(p.tp)} hint={`recall ${fmtPct(p.recall)}`} />
        <Stat label="Hit rate" value={fmtPct(p.precision)} hint="of visits find theft" />
        <Stat label="Wasted visits" value={fmtInt(p.fp)} hint={`${fmtInt(p.fn)} thefts missed`} />
      </section>

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-[1.3fr_1fr]">
        <Card label="Precision and recall" className="flex flex-col gap-3 p-[22px]">
          <CardHeader title="Trade-off curve" subtitle="Click the chart to jump to a threshold."
            action={<Legend items={[{ color: "#2346C8", label: "Hit rate (precision)", line: true }, { color: "#E0873A", label: "Thefts caught (recall)", line: true }]} />} />
          <TradeoffChart points={points} index={index} onPick={setIndex} trainedThreshold={trained} />
        </Card>

        <Card label="Confusion matrix" className="flex flex-col gap-4 p-[22px]">
          <CardHeader title="What happens on the ground" />
          <ConfusionGrid tp={p.tp} fp={p.fp} fn={p.fn} tn={p.tn} />
          <div className="mt-auto flex flex-col gap-2.5">
            {publish.isError ? <ErrorState title="Could not publish" message={apiErrorMessage(publish.error)} /> : null}
            {publish.isSuccess && isInService ? (
              <p role="status" className="m-0 text-[13px] text-cobalt-ink">Published. Scoring now flags {fmtInt(p.tp + p.fp)} customers; new flags open cases.</p>
            ) : null}
            <Button variant="primary" disabled={isInService} busy={publish.isPending} onClick={() => publish.mutate(p.threshold)}>
              {isInService ? `τ ${p.threshold.toFixed(2)} is in service` : `Publish τ ${p.threshold.toFixed(2)} to scoring`}
            </Button>
            {Math.abs(inService - trained) > 0.005 ? (
              <Button variant="ghost" busy={publish.isPending} onClick={() => publish.mutate(null)}>Return to trained threshold ({trained.toFixed(3)})</Button>
            ) : null}
          </div>
        </Card>
      </div>
    </>
  );
}
