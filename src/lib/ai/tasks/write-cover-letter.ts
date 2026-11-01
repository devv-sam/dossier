import { z } from "zod";

import type { Tone } from "@/generated/prisma/enums";
import { SIGN_OFF_UNIT, unitsSchema, type Extracted, type Match, type Unit } from "@/lib/application-data";
import type { ProfileData } from "@/lib/profile";
import { FRAME_LABELS, isFrameUnit } from "@/lib/letter-frame";
import { TONES } from "@/lib/tones";

import { guardSources, guardUnits } from "../fact-guard";

import {
  applicationFacts,
  candidateFacts,
  COVER_LETTER_APPROACH,
  CREATIVITY,
  jobForPrompt,
  languageRule,
  NO_EM_DASHES,
  profileForPrompt,
  system,
  truthfulness,
  UNTRUSTED_POSTING,
  type Task,
} from "./prompt-parts";

export const WRITE_COVER_LETTER_VERSION = "write-cover-letter@6";

const answerSchema = z.object({
  recipient: z.string(),
  greeting: z.string(),
  paragraphs: z.array(z.object({ label: z.string(), text: z.string() })),
  signOff: z.string(),
});

const INSTRUCTIONS = `You write a candidate's cover letter for one job, as JSON.

- recipient: the address block, one line per line: who it's for (the company's hiring team, or the contact person the posting names, at the company), then the place if the job has one. Usual for a letter in its language, e.g. "Ledgerly · Hiring team\\nAmsterdam".
- greeting: the salutation before the first paragraph, usual for a letter in its language, with its punctuation: the contact person by name when the posting names one, else the hiring team, e.g. "Dear Ledgerly team,".
- paragraphs: 3 short paragraphs, about 120–170 words together, never more than 200. Open with why this role and this company in one or two sentences, then where the candidate has solved a similar problem before (one specific story), and close in one or two sentences by inviting a conversation, with where the candidate is based if it matters for the job. No greeting, sign-off or name in them: those are separate.
- label: a short English name for each paragraph's purpose, e.g. "Why this role", "Where I've solved it", "Closing".
- Where the job asks for something the profile doesn't have ("gap"), don't claim it. If it matters, say so honestly in a sentence and show how the candidate would close the gap.
- signOff: the closing line before the name, matching the tone, with its punctuation (e.g. "Warm regards,").
- The sender's name, contact details and the date are added from the profile; don't write them.`;

export type CoverLetterInput = {
  profile: ProfileData;
  extracted: Extracted;
  match: Match | null;
  tone: Tone;
  /** What the candidate tells the AI for this application (APP-15). */
  facts: string[];
  /** The posting as pasted, for the fact guard. */
  posting: string;
};

/**
 * APP-10: the letter's recipient and greeting (units "recipient" and
 * "greeting", E8-11), its paragraphs as units, then its sign-off (unit
 * "sign-off"), which matches the tone. The sender, date and signature come
 * from the profile when the application is created (src/lib/letter-frame.ts).
 */
export const writeCoverLetter: Task<CoverLetterInput, Unit[]> = async (provider, { profile, extracted, match, tone, facts, posting }, options) => {
  const instructions = system(
    INSTRUCTIONS,
    COVER_LETTER_APPROACH,
    `Tone: ${TONES[tone].label}. ${TONES[tone].description}`,
    truthfulness(options.strict),
    candidateFacts(profile, facts),
    CREATIVITY[options.creativity],
    languageRule(options.language),
    UNTRUSTED_POSTING,
    NO_EM_DASHES,
  );
  const request = {
    role: "user" as const,
    content: [`The candidate's profile:\n${profileForPrompt(profile)}`, `The job:\n${jobForPrompt(extracted, match)}`, applicationFacts(facts)]
      .filter(Boolean)
      .join("\n\n"),
  };
  const ask = (messages: { role: "user" | "assistant"; content: string }[]) =>
    provider.complete({ system: instructions, messages, schema: answerSchema, schemaName: "cover_letter", creativity: options.creativity, signal: options.signal });

  // English letters get the tone's own sign-off; other languages keep the model's.
  const language = options.language ?? extracted.language ?? "English";
  const english = /^english$/i.test(language);
  // Small models ignore a list of words to avoid, so swap the common ones for plain words.
  const cleaned = (written: z.infer<typeof answerSchema>) => (english ? { ...written, paragraphs: written.paragraphs.map((paragraph) => ({ ...paragraph, text: plainWords(paragraph.text) })) } : written);

  let answer = cleaned(await ask([request]));
  // What's left gets up to two rewrites, each kept only if it's no worse.
  if (english) {
    let problems = voiceProblems(answer.paragraphs);
    for (let attempt = 0; attempt < MAX_REWRITES && problems.score > 0; attempt++) {
      const retry = cleaned(await ask([request, { role: "assistant", content: JSON.stringify(answer) }, { role: "user", content: rewriteRequest(problems) }]));
      const next = voiceProblems(retry.paragraphs);
      if (next.score <= problems.score) {
        answer = retry;
        problems = next;
      }
    }
  }
  const signOff = /^english$/i.test(language) || !answer.signOff.trim() ? TONES[tone].signOff : answer.signOff.trim();
  const units = unitsSchema.parse([
    // Left out when empty: the English ones fill in (src/lib/letter-frame.ts).
    ...(answer.recipient.trim() ? [{ id: "recipient", label: FRAME_LABELS.recipient, text: tidyLines(answer.recipient) }] : []),
    ...(answer.greeting.trim() ? [{ id: "greeting", label: FRAME_LABELS.greeting, text: answer.greeting.trim() }] : []),
    ...withoutFrame(answer.paragraphs, signOff)
      .filter((paragraph) => paragraph.text.trim())
      .map((paragraph, index) => ({ id: `paragraph:${index + 1}`, label: paragraph.label.trim() || `Paragraph ${index + 1}`, text: paragraph.text.trim() })),
    { id: SIGN_OFF_UNIT, label: "Sign-off", text: signOff },
  ]);
  return {
    // The recipient and greeting aren't claims about the candidate: only the paragraphs are checked.
    data: options.strict ? guardUnits(units, guardSources(profile, extracted, posting, facts)).map((unit) => (isFrameUnit(unit.id) ? units.find((entry) => entry.id === unit.id)! : unit)) : units,
    promptVersion: WRITE_COVER_LETTER_VERSION,
  };
};

const MAX_WORDS = 190;
const MAX_REWRITES = 2;

/** Words and phrases that mark a letter as machine-written (see COVER_LETTER_APPROACH). */
const STOCK_WORDS =
  /\b(?:excit\w*|thrill\w*|eager\w*|passion\w*|aligns? (?:perfectly|with)|great fit|leverag\w*|utiliz\w*|empower\w*|foster\w*|cultivat\w*|facilitat\w*|streamlin\w*|seamless\w*|robust|holistic|innovative|cutting-edge|user-centric|journey|landscape|testament|furthermore|moreover|additionally|significantly|crucial|essential|resonates?|delve|revolutioniz\w*|I believe|I am confident|I'm confident|boundaries)\b/gi;

const VERB_ENDINGS: Record<string, string> = { e: "use", es: "uses", ed: "used", ing: "using" };
const useVerb = (_match: string, ending: string) => VERB_ENDINGS[ending.toLowerCase()] ?? "use";

/** Stock words with a plain swap that keeps the sentence working, in the order they're tried. */
const PLAIN_SWAPS: [RegExp, string | ((match: string, ending: string) => string)][] = [
  [/\b(?:excited|thrilled|eager) to\b/gi, "glad to"],
  [/\b(?:excited|thrilled|eager|passionate) (?:about|by)\b/gi, "interested in"],
  [/\bpassion for\b/gi, "interest in"],
  [/\b(?:excited|thrilled)\b/gi, "glad"],
  [/\beager\b/gi, "keen"],
  [/\bpassionate\b/gi, "keen"],
  [/\bseamless(ly)?\b/gi, "smooth$1"],
  [/\brobust\b/gi, "solid"],
  [/\b(?:innovative|cutting-edge)\b/gi, "new"],
  [/\buser-centric\b/gi, "user-focused"],
  [/\bholistic\b/gi, "whole"],
  [/\bleverag(e|es|ed|ing)\b/gi, useVerb],
  [/\butiliz(e|es|ed|ing)\b/gi, useVerb],
  [/\baligns? (?:perfectly )?with\b/gi, "matches"],
  [/\bgreat fit\b/gi, "good match"],
  [/\b(?:crucial|essential)\b/gi, "important"],
  [/\bsignificantly /gi, ""],
];

/** A letter's text with the stock words in PLAIN_SWAPS replaced, and "Additionally," style openers dropped. */
export function plainWords(text: string): string {
  let result = text;
  for (const [pattern, swap] of PLAIN_SWAPS) result = typeof swap === "string" ? result.replace(pattern, swap) : result.replace(pattern, swap);
  return result
    .replace(/(^|[.!?]\s+)(?:Additionally|Furthermore|Moreover),?\s+([A-Za-z])/g, (_, lead: string, letter: string) => lead + letter.toUpperCase())
    .replace(/\s{2,}/g, " ");
}

/** What's wrong with a letter's voice: the stock words in it, whether it runs long, and a score to compare drafts by. */
export function voiceProblems(paragraphs: { text: string }[]): { words: string[]; wordCount: number; tooLong: boolean; score: number } {
  const text = paragraphs.map((paragraph) => paragraph.text).join(" ");
  const words = [...new Set((text.match(STOCK_WORDS) ?? []).map((word) => word.toLowerCase()))];
  const wordCount = text.split(/\s+/).filter(Boolean).length;
  const tooLong = wordCount > MAX_WORDS;
  return { words, wordCount, tooLong, score: words.length + (tooLong ? 2 : 0) };
}

function rewriteRequest({ words, wordCount, tooLong }: ReturnType<typeof voiceProblems>): string {
  return [
    "Rewrite the letter with the same facts and the same JSON shape.",
    words.length > 0
      ? `Don't use these words or phrases anywhere: ${words.join(", ")}. Delete any sentence that only says how the candidate feels about the role, and say the plain thing instead.`
      : "",
    tooLong ? `It is ${wordCount} words. Cut it to 150 words or fewer: three short paragraphs.` : "",
  ]
    .filter(Boolean)
    .join(" ");
}

/** An address block's lines, trimmed, without empty ones. */
const tidyLines = (text: string) =>
  text
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean)
    .join("\n");

const GREETING = /^\s*(?:dear|hi|hello|hey|good (?:morning|afternoon|day))\b[^\n]{0,60}[,!:]?\s*(?:\n|$)/i;

/**
 * The renderer adds the greeting, sign-off and name (APP-10); models add them
 * anyway sometimes. Takes a greeting line off the first paragraph and a
 * sign-off (with the name after it) off the last.
 */
export function withoutFrame<T extends { text: string }>(paragraphs: T[], signOff: string): T[] {
  if (paragraphs.length === 0) return paragraphs;
  const result = paragraphs.map((paragraph) => ({ ...paragraph }));
  result[0].text = result[0].text.replace(GREETING, "");
  const last = result.at(-1)!;
  const closings = [signOff, "Warm regards", "Kind regards", "Best regards", "Best", "Sincerely", "Yours sincerely", "Regards"];
  for (const closing of closings) {
    const escaped = closing.replace(/[,.]$/, "").replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    // On its own line, or right after the last sentence, maybe followed by a name.
    last.text = last.text.replace(new RegExp(`(?:\\n+|(?<=[.!?])[ \\t]+)\\s*${escaped}[,.]?(?:\\s*\\n[^\\n]{0,60})?\\s*$`, "i"), "").trim();
  }
  return result;
}
