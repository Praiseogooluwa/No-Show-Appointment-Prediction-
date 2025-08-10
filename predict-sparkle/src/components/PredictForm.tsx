import { useMemo, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useToast } from "@/hooks/use-toast";

interface ApiResult {
  prediction: string;
  no_show_probability: number; // 0-1
  confidence: string;
  recommendation: string;
  risk_factors: Record<string, string>;
}

const API_BASE = "https://no-show-appointment-prediction.onrender.com"; // Change if your API runs elsewhere

function toLocalInputValue(date: Date) {
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}T${pad(date.getHours())}:${pad(date.getMinutes())}`;
}

export const PredictForm = () => {
  const { toast } = useToast();
  const now = useMemo(() => new Date(), []);
  const week = useMemo(() => {
    const d = new Date();
    d.setDate(d.getDate() + 7);
    return d;
  }, []);

  const [gender, setGender] = useState<string>("");
  const [age, setAge] = useState<number>(30);
  const [scheduledDay, setScheduledDay] = useState<string>(toLocalInputValue(now));
  const [appointmentDay, setAppointmentDay] = useState<string>(toLocalInputValue(week));
  const [scholarship, setScholarship] = useState<number>(0);
  const [hipertension, setHipertension] = useState<number>(0);
  const [diabetes, setDiabetes] = useState<number>(0);
  const [alcoholism, setAlcoholism] = useState<number>(0);
  const [handcap, setHandcap] = useState<number>(0);
  const [smsReceived, setSmsReceived] = useState<number>(1);

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ApiResult | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setResult(null);

    // Basic validation
    if (!gender) {
      toast({ title: "Missing gender", description: "Please select a gender." });
      setLoading(false);
      return;
    }

    try {
      const payload = {
        gender,
        age: Number(age),
        scheduled_day: new Date(scheduledDay).toISOString(),
        appointment_day: new Date(appointmentDay).toISOString(),
        scholarship: Number(scholarship),
        hipertension: Number(hipertension),
        diabetes: Number(diabetes),
        alcoholism: Number(alcoholism),
        handcap: Number(handcap),
        sms_received: Number(smsReceived),
      };

      const res = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const data = await res.json();

      if (!res.ok) {
        throw new Error(data?.detail || "Request failed");
      }

      // Guard against NaN/undefined
      const prob = Number(data?.no_show_probability);
      setResult({
        prediction: String(data?.prediction ?? "Unknown"),
        no_show_probability: Number.isFinite(prob) ? prob : 0,
        confidence: String(data?.confidence ?? "Unknown"),
        recommendation: String(data?.recommendation ?? ""),
        risk_factors: (data?.risk_factors as Record<string, string>) ?? {},
      });
    } catch (err: any) {
      toast({
        title: "Prediction failed",
        description: err?.message || "Could not reach the API. Ensure it is running.",
      });
    } finally {
      setLoading(false);
    }
  };

  const isNoShow = result?.prediction === "No-Show";
  const percent = result ? Math.round((result.no_show_probability || 0) * 1000) / 10 : 0;

  return (
    <div className="grid gap-6 lg:grid-cols-2">
      <Card className="border border-input/60 backdrop-blur supports-[backdrop-filter]:bg-background/70">
        <CardHeader>
          <CardTitle>Patient & Appointment Details</CardTitle>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="grid gap-4 animate-in fade-in-50">
            <div className="grid gap-4 sm:grid-cols-2">
              <div className="grid gap-2">
                <Label htmlFor="gender">Gender</Label>
                <select
                  id="gender"
                  value={gender}
                  onChange={(e) => setGender(e.target.value)}
                  className="h-10 rounded-md border border-input bg-background px-3 text-sm text-foreground shadow-sm outline-none ring-offset-background focus-visible:ring-2 focus-visible:ring-ring"
                  required
                >
                  <option value="">Select Gender</option>
                  <option value="F">Female</option>
                  <option value="M">Male</option>
                </select>
              </div>

              <div className="grid gap-2">
                <Label htmlFor="age">Age</Label>
                <Input id="age" type="number" min={0} max={120} value={age}
                       onChange={(e) => setAge(Number(e.target.value))} required />
              </div>

              <div className="grid gap-2">
                <Label htmlFor="scheduled">Scheduled Day</Label>
                <Input id="scheduled" type="datetime-local" value={scheduledDay}
                       onChange={(e) => setScheduledDay(e.target.value)} required />
              </div>

              <div className="grid gap-2">
                <Label htmlFor="appointment">Appointment Day</Label>
                <Input id="appointment" type="datetime-local" value={appointmentDay}
                       onChange={(e) => setAppointmentDay(e.target.value)} required />
              </div>

              <div className="grid gap-2">
                <Label htmlFor="scholarship">Scholarship</Label>
                <select id="scholarship" value={scholarship}
                        onChange={(e) => setScholarship(Number(e.target.value))}
                        className="h-10 rounded-md border border-input bg-background px-3 text-sm text-foreground shadow-sm outline-none ring-offset-background focus-visible:ring-2 focus-visible:ring-ring">
                  <option value={0}>No</option>
                  <option value={1}>Yes</option>
                </select>
              </div>

              <div className="grid gap-2">
                <Label htmlFor="hipertension">Hypertension</Label>
                <select id="hipertension" value={hipertension}
                        onChange={(e) => setHipertension(Number(e.target.value))}
                        className="h-10 rounded-md border border-input bg-background px-3 text-sm text-foreground shadow-sm outline-none ring-offset-background focus-visible:ring-2 focus-visible:ring-ring">
                  <option value={0}>No</option>
                  <option value={1}>Yes</option>
                </select>
              </div>

              <div className="grid gap-2">
                <Label htmlFor="diabetes">Diabetes</Label>
                <select id="diabetes" value={diabetes}
                        onChange={(e) => setDiabetes(Number(e.target.value))}
                        className="h-10 rounded-md border border-input bg-background px-3 text-sm text-foreground shadow-sm outline-none ring-offset-background focus-visible:ring-2 focus-visible:ring-ring">
                  <option value={0}>No</option>
                  <option value={1}>Yes</option>
                </select>
              </div>

              <div className="grid gap-2">
                <Label htmlFor="alcoholism">Alcoholism</Label>
                <select id="alcoholism" value={alcoholism}
                        onChange={(e) => setAlcoholism(Number(e.target.value))}
                        className="h-10 rounded-md border border-input bg-background px-3 text-sm text-foreground shadow-sm outline-none ring-offset-background focus-visible:ring-2 focus-visible:ring-ring">
                  <option value={0}>No</option>
                  <option value={1}>Yes</option>
                </select>
              </div>

              <div className="grid gap-2">
                <Label htmlFor="handcap">Handicap Level</Label>
                <select id="handcap" value={handcap}
                        onChange={(e) => setHandcap(Number(e.target.value))}
                        className="h-10 rounded-md border border-input bg-background px-3 text-sm text-foreground shadow-sm outline-none ring-offset-background focus-visible:ring-2 focus-visible:ring-ring">
                  {[0,1,2,3,4].map((n) => (
                    <option key={n} value={n}>{n === 0 ? "None" : `Level ${n}`}</option>
                  ))}
                </select>
              </div>

              <div className="grid gap-2">
                <Label htmlFor="sms">SMS Reminder Sent</Label>
                <select id="sms" value={smsReceived}
                        onChange={(e) => setSmsReceived(Number(e.target.value))}
                        className="h-10 rounded-md border border-input bg-background px-3 text-sm text-foreground shadow-sm outline-none ring-offset-background focus-visible:ring-2 focus-visible:ring-ring">
                  <option value={0}>No</option>
                  <option value={1}>Yes</option>
                </select>
              </div>
            </div>

            <Button type="submit" disabled={loading} className="mt-2">
              {loading ? "Predicting..." : "Predict No-Show"}
            </Button>
          </form>
        </CardContent>
      </Card>

      <Card className="border border-input/60 backdrop-blur supports-[backdrop-filter]:bg-background/70">
        <CardHeader>
          <CardTitle>Prediction Results</CardTitle>
        </CardHeader>
        <CardContent>
          {!result && (
            <p className="text-sm text-muted-foreground">Fill the form and click Predict to see results.</p>
          )}

          {result && (
            <div className="space-y-4 animate-in fade-in-50">
              <div>
                <p className="text-sm">Prediction</p>
                <p className="text-2xl font-semibold">{result.prediction}</p>
              </div>
              <div>
                <p className="text-sm">No-Show Probability</p>
                <p className={`text-2xl font-semibold ${isNoShow ? "text-destructive" : "text-primary"}`}>
                  {percent.toFixed(1)}%
                </p>
              </div>
              <div>
                <p className="text-sm">Confidence</p>
                <p className="font-medium">{result.confidence}</p>
              </div>
              {result.recommendation && (
                <div>
                  <p className="text-sm">Recommendation</p>
                  <p className="font-medium">{result.recommendation}</p>
                </div>
              )}

              {result.risk_factors && Object.keys(result.risk_factors).length > 0 && (
                <div>
                  <p className="text-sm mb-2">Risk Factors</p>
                  <div className="grid gap-2">
                    {Object.entries(result.risk_factors).map(([k, v]) => (
                      <div key={k} className="rounded-md border border-input px-3 py-2 text-sm">
                        <span className="font-medium capitalize">{k.replace(/_/g, " ")}: </span>
                        <span>{v}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
};

export default PredictForm;
