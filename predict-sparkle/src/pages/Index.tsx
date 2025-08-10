import PredictForm from "@/components/PredictForm";

const Index = () => {
  return (
    <main className="min-h-screen bg-background relative">
      <section className="relative isolate overflow-hidden">
        <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(ellipse_at_top,hsl(var(--primary)/0.12),transparent_60%)]" />
        <div className="container py-12 md:py-16">
          <header className="mb-8 text-center">
            <h1 className="text-4xl md:text-5xl font-bold tracking-tight">Hospital No-Show Prediction Dashboard</h1>
            <p className="mt-3 text-muted-foreground max-w-2xl mx-auto">
              Enter patient and appointment details to predict attendance and understand risk factors.
            </p>
          </header>

          <PredictForm />
        </div>
      </section>
    </main>
  );
};

export default Index;
