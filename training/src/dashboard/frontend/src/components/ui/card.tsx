import * as React from "react";

import { cn } from "@/lib/utils";

function Card({ className, ...props }: React.ComponentProps<"div">) {
  return (
    <div
      data-slot="card"
      className={cn(
        "bg-card text-card-foreground flex flex-col rounded-xl border shadow-sm",
        className,
      )}
      {...props}
    />
  );
}

function CardHeader({ className, ...props }: React.ComponentProps<"div">) {
  return (
    <div
      data-slot="card-header"
      className={cn(
        "@container/card-header grid auto-rows-min grid-rows-[auto_auto] items-start gap-2 px-6 has-data-[slot=card-action]:grid-cols-[1fr_auto] [.border-b]:pb-6",
        className,
      )}
      {...props}
    />
  );
}

function CardTitle({ className, ...props }: React.ComponentProps<"div">) {
  return (
    <div
      data-slot="card-title"
      className={cn("leading-none font-semibold", className)}
      {...props}
    />
  );
}

function CardDescription({ className, ...props }: React.ComponentProps<"div">) {
  return (
    <div
      data-slot="card-description"
      className={cn("text-muted-foreground text-sm", className)}
      {...props}
    />
  );
}

function CardAction({ className, ...props }: React.ComponentProps<"div">) {
  return (
    <div
      data-slot="card-action"
      className={cn(
        "col-start-2 row-span-2 row-start-1 self-start justify-self-end",
        className,
      )}
      {...props}
    />
  );
}

function CardContent({ className, ...props }: React.ComponentProps<"div">) {
  return (
    <div
      data-slot="card-content"
      className={cn("px-6", className)}
      {...props}
    />
  );
}

function CardFooter({ className, ...props }: React.ComponentProps<"div">) {
  return (
    <div
      data-slot="card-footer"
      className={cn("flex items-center px-6 [.border-t]:pt-6", className)}
      {...props}
    />
  );
}

// Compact panel variant (px-3) — used in run-detail panels and sidebar cards
function PanelCardHeader({ className, ...props }: React.ComponentProps<"div">) {
  return <CardHeader className={cn("px-3 pb-2 pt-4", className)} {...props} />;
}

function PanelCardTitle({ className, ...props }: React.ComponentProps<"div">) {
  return <CardTitle className={cn("flex items-center gap-2 text-sm font-medium", className)} {...props} />;
}

function PanelCardContent({ className, ...props }: React.ComponentProps<"div">) {
  return <CardContent className={cn("px-3 pb-3", className)} {...props} />;
}

// Section variant (px-4) — used in launcher / setup panels
function SectionCardHeader({ className, ...props }: React.ComponentProps<"div">) {
  return <CardHeader className={cn("px-4 pb-2 pt-4", className)} {...props} />;
}

function SectionCardTitle({ className, ...props }: React.ComponentProps<"div">) {
  return <CardTitle className={cn("flex items-center gap-2 text-sm font-medium", className)} {...props} />;
}

function SectionCardContent({ className, ...props }: React.ComponentProps<"div">) {
  return <CardContent className={cn("px-4 pb-4", className)} {...props} />;
}

export {
  Card,
  CardHeader,
  CardFooter,
  CardTitle,
  CardAction,
  CardDescription,
  CardContent,
  PanelCardHeader,
  PanelCardTitle,
  PanelCardContent,
  SectionCardHeader,
  SectionCardTitle,
  SectionCardContent,
};
