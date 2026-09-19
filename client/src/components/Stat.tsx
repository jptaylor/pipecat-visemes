/** One labeled value in a stats row. */
export function Stat({
  label,
  value,
  title,
  warn = false,
}: {
  label: string;
  value: string;
  title?: string;
  warn?: boolean;
}) {
  return (
    <span className="stat" title={title}>
      <span className="stat-label">{label}</span>
      <span className={warn ? "stat-value stat-warn" : "stat-value"}>{value}</span>
    </span>
  );
}
