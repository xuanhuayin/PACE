from pace.paths import PROJECT_ROOT
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = PROJECT_ROOT


import pace.construction.ifeval as G
import tests.fixtures.models as T3

FAILS = []
VERBOSE = "-v" in sys.argv


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


def has(text, tag, **plan_):
    return tag in G.g4_issues(text, plan_)


P1 = "The committee reviewed the proposal carefully and found that the budget was realistic for the first year."
P2 = "Several members asked for a clearer timeline, so the chair promised a revised schedule by Friday."
P3 = "Finally the group agreed to meet again next month to vote on the amended version of the plan."
P4 = "A short note on costs was added to the appendix for reference by the finance office."
PARA = dict(kind="paragraphs", n=3, sep="\n***\n", block_prefix="", cut="\n\n")
TWO = dict(kind="two", n=2, sep="\n******\n", block_prefix="", cut="******")
BUL = dict(kind="bullets", n=2, sep="\n", block_prefix="* ", cut="\n")


def test_units():
    check(
        has(P1 + "\n\n1.\n\n2.\n\n3. " + P2, "A:skeleton"),
        "A+ bare list numbers (empty numbered skeleton) flagged",
    )
    check(
        not has(
            "1. Mix the flour with the sugar.\n2. Add the water slowly.\n3. Bake it for an hour.",
            "A:skeleton",
        ),
        "A- a real numbered list is not a skeleton",
    )
    check(
        has("Structure:\n\n2. Harmony:\n\n3. Instrumentation:\n\n4. Technique:", "A:skeleton"),
        "A+ label-only items without content flagged",
    )
    check(
        has(P1 + "\n***\nSECTION0\n***\n" + P2, "A:empty_block", blocks=PARA),
        "A+ junk marker block between separators flagged",
    )
    check(
        not has(P1 + "\n***\n" + P2 + "\n***\n" + P3, "A:empty_block", blocks=PARA),
        "A- three real paragraphs pass",
    )
    check(
        has(P1 + "\n***\n" + P1 + "\n***\n" + P3, "A:dup_block", blocks=PARA),
        "A+ duplicated paragraph flagged",
    )
    check(
        has("CERTAINLY HERE ARE SOME RECOMMENDATIONS FOR YOUR ESSAY:\n1.", "A:fragment_end"),
        "A+ dangling bare list number at the end flagged",
    )
    check(
        has(
            "I'll be free stealing every car\nI'll leave on bail and I'll take ground I I",
            "A:fragment_end",
        ),
        "A+ single-letter degenerate tail flagged",
    )
    check(not G.g4_issues(P1 + "\n\nBest regards,\nJohn", {}), "A- a sign-off ending is clean")
    rap = "They'll try my bail, they'll do it, no doubt\nI'll be leaving the man's bail, I'll make it bold\n***I'll be on the prowl he'll make them scatter\nI'll be on every corner"
    check(has(rap, "B:inline_sep", blocks=PARA), "B+ separator glued mid-stanza flagged")
    check(
        not has(
            "A solid plan matters.***\n\nBefore you begin, reflect on your goals.\n\n***As I watched her walk, I smiled.",
            "B:inline_sep",
            blocks=PARA,
        ),
        "B- separators after a sentence end / blank line pass (LLaMA-3 style)",
    )
    lst = "* In the second year it grows six inches.\n* In the third year it grows again.\n*** In 1 year the colt will be tall.\n* In 2 years it will be taller."
    check(has(lst, "B:sep_mid_list", blocks=PARA), "B+ separator inside a list flagged")
    check(
        has(
            "The tea was poured and\n***\nthe guests sat down to talk.",
            "B:sep_mid_sentence",
            blocks=PARA,
        ),
        "B+ separator splitting a sentence flagged",
    )
    pre = "There are two names people call God: Allah and Yahweh. Here is the formatted version of the two responses:\n******\nResponse 1: Islam\nAllah is the name of God in Islam.\n\nResponse 2: Judaism\nYahweh is the name of God in Judaism."
    t = G.g4_issues(pre, dict(blocks=TWO))
    check(
        "B:two_preamble" in t and "B:two_merged" in t,
        f"B+ two_responses preamble split + both answers in one block flagged ({sorted(t)})",
    )
    check(
        not G.g4_issues(P1 + " " + P2 + "\n******\n" + P3 + " " + P4, dict(blocks=TWO)),
        "B- two full answers pass",
    )
    check(
        has(
            "* High-quality materials for long-lasting use 2 Sleek and minimalist design\n* Compatibility with many devices",
            "B:bullet_merged",
            blocks=BUL,
        ),
        "B+ two items squashed into one bullet flagged",
    )
    check(
        not has("* Spot\n* Max\n* Luna", "A:empty_block", blocks=dict(BUL, n=3)),
        "B- one-word bullets (names) pass",
    )
    letter = "Thank you for your time.\nPlease review my attached << and I welcome the opportunity to discuss my application >> and for considering me."
    check(
        has(letter, "C:title_inline", title=True), "C+ << >> around a mid-sentence fragment flagged"
    )
    check(has("<<title>>\n\n" + P1, "C:title_filler", title=True), "C+ literal <<title>> flagged")
    echo = "Write a plot. Include a title wrapped in double angular brackets, i.e. <<title>>.\n\n"
    check(
        has(echo + P1 + " " + P2, "C:title_echo_only", title=True, prefix=echo),
        "C+ title only inside the echoed prompt flagged",
    )
    check(
        not G.g4_issues("<<The Fingerprint Swap>>\n\n" + P1, dict(title=True)),
        "C- a real title line passes",
    )
    check(
        not G.g4_issues(
            "Here are the names:\n\n<<Brand>> <<Elixir Revive>> and <<Vitality Pulse>>",
            dict(title=True),
        ),
        "C- inline name titles pass",
    )
    ph = "Dear [Name],\n[I just wanted to let you know that the boots] ordered for [Event]/[asion] were snug. I[m happy to help].\n[Subject] Boot [ed]"
    check(
        has(ph, "D:ph_fake", placeholders=6),
        "D+ brackets around sentences / word fragments flagged",
    )
    check(
        not G.g4_issues(
            "Dear [Name], we will meet at [Address] on [Date] to sign the [Contract Type].",
            dict(placeholders=4),
        ),
        "D- real placeholders pass",
    )
    hl = "**Light** Trust is key.\n**Highlight** A healthy sex life can make or break a marriage.\n*Highlighted section Part 3*\n\nMore text."
    check(has(hl, "E:hl_fake", highlights=3), "E+ meta-label highlights flagged")
    check(
        has("Prices rose 5 *. * fell and *,* returned.", "E:hl_fake", highlights=2),
        "E+ punctuation-fragment highlights flagged",
    )
    check(
        not G.g4_issues(
            "*The Role of Resilience in Mythology*Peterson argues that *we grow through trials* and *stories matter*.",
            dict(highlights=3),
        ),
        "E- real highlights (also glued to the next word) pass",
    )
    body = "Navigable waterways provide a safe and efficient means of transportation for goods and services. They are essential for global trade and commerce across many regions. Boats rely on locks, buoys and channels to travel safely through large bodies of water."
    rec = (
        body
        + "\n\n"
        + body.replace("Navigable", "navigable").replace("They are", "they are truly")
        + " "
        + body.replace("safe", "safe and reliable")
    )
    check(
        has(rec, "F:recycled", min_words=G.count_words(rec) - 5),
        "F+ word minimum reached by recycled sentences flagged",
    )
    check(
        not has(P1 + " " + P2 + " " + P3 + " " + P4, "F:recycled", min_words=60),
        "F- distinct sentences pass",
    )
    check(
        has(
            body
            + "\n\nIn conclusion, waterways matter.\n\n"
            + P1
            + "\n\nIn summary, waterways matter a lot.",
            "F:double_conclusion",
            min_words=50,
        ),
        "F+ second conclusion flagged",
    )
    check(
        has("Here is my list of facts.\n1.\n2.\n3.\n4.\n5.", "F:sent_fake", min_sent=5),
        "F+ sentence minimum reached by bare list numbers flagged",
    )
    check(
        not has(
            " ".join([P1, P2, P3, P4, P1.replace("committee", "board")]), "F:sent_fake", min_sent=5
        ),
        "F- five real sentences pass",
    )
    kr = lambda k, n=1: [dict(kw=k, relation="at least", n=n)]
    check(
        has(
            "The peel protects the fruit from damage. disappointed================",
            "G:kw_fake",
            kw_req=kr("disappointed"),
        ),
        "G+ keyword glued to a symbol run flagged",
    )
    check(
        has(
            "This would be an extremely high overd dose for anyone.", "G:kw_fake", kw_req=kr("dose")
        ),
        "G+ split word ('overd dose') flagged",
    )
    check(
        not G.g4_issues(
            "This would be an overdose for anyone who drinks it.", dict(kw_req=kr("dose"))
        ),
        "G- keyword inside a real word ('overdose') passes",
    )
    check(
        has(
            "NIKE IS KNOWN FOR STYLESNEAKERS AND ITSNEAKERLETE DESIGNS.",
            "G:kw_fake",
            kw_req=kr("sneaker", 2),
        ),
        "G+ keyword glued inside non-words flagged",
    )
    check(
        has(
            'Your task is to write a limerick that uses the word "grow" and the word "intern".',
            "G:kw_fake",
            kw_req=kr("grow"),
        ),
        "G+ keyword only in a meta-statement flagged",
    )
    e2 = "Include the words intern and grow.\n\n"
    check(
        has(
            e2 + "Hannah poured coffee every single morning.",
            "G:kw_fake",
            kw_req=kr("grow"),
            prefix=e2,
        ),
        "G+ keyword only in the echoed prompt flagged",
    )
    check(
        not G.g4_issues(
            "I was disappointed by the ending of the film.", dict(kw_req=kr("disappointed"))
        ),
        "G- keyword in a normal sentence passes",
    )
    lt = [dict(letter="t", relation="less than", n=1)]
    check(
        has(
            "Which kind of sofa is a cheserfield? Ours has a velvel cover and a backress for comfort.",
            "G:letter_mangled",
            letters=lt,
        ),
        "G+ words mangled to avoid a letter flagged",
    )
    check(
        not has(
            "Which kind of sofa is a large one? Ours has a velvet cover and a soft back for comfort.",
            "G:letter_mangled",
            letters=[dict(letter="z", relation="less than", n=1)],
        ),
        "G- normal words pass the letter check",
    )
    rid = "What has a dome but no doors? What am I?\n\n"
    check(
        has(rid + "What am I?_ You and I I I Hi Hi I I I I", "H:echo_soup", prefix=rid),
        "H+ echo followed by token soup flagged",
    )
    check(
        not has(
            rid + "You are the Capitol building, of course, and you never pay rent.",
            "H:echo_soup",
            prefix=rid,
        ),
        "H- echo + real answer passes",
    )
    meta = "Part #1: Repeat the request without change.\n\nPart #2: Give your answer.\n\nYour task is to write a limerick about Hannah and her internship."
    check(has(meta, "H:meta_answer"), "H+ meta answer (talks about the task) flagged")
    ep = "Is there anything else I can help with?"
    check(
        has(
            'The sentiment is unclear. In any case, the last sentence of my response will be "'
            + ep
            + '"',
            "H:end_meta",
            end_phrase=ep,
        ),
        "H+ end phrase quoted in a meta-statement flagged",
    )
    check(
        not G.g4_issues('The tallest tower is in Dubai. "' + ep + '"', dict(end_phrase=ep)),
        "H- quoted end phrase after a real sentence passes",
    )
    ps = G.g4_issues(
        "\n\nP.P.S Hey sweetie! Here are 10 possible names for a new baby boy:",
        dict(postscript="P.P.S"),
    )
    check(
        "H:ps_dangling" in ps and "H:ps_only" in ps,
        f"H+ P.S. introducing content that never appears flagged ({sorted(ps)})",
    )
    check(
        not G.g4_issues(P1 + " " + P2 + "\n\nP.S. Happy listening!", dict(postscript="P.S.")),
        "H- short real P.S. passes",
    )
    check(
        has("I can give you some recommendations:", "H:dangling_colon"),
        "H+ final line promising content flagged",
    )
    check(
        has("ǎ a I I I a a a a***ó a a a a a a a a a a and and and", "I:letter_run"),
        "I+ single-letter token soup flagged",
    )
    junk = "The navable freeways ar teh bestest trnsport fr goods an servces becuz thay allw botes to travl acros larje bodys of watr safly and quikly withot encountring problms or delys at al"
    check(has(junk, "I:pseudo_words"), "I+ English pseudo-words flagged")
    check(not has(P1 + " " + P2 + " " + P3, "I:pseudo_words"), "I- normal English passes")
    check(
        has("everything it has to offer. disappointed================---", "I:symbol_run"),
        "I+ symbol run glued to a word flagged",
    )
    soup = " ".join(("घऱऱ" + s for s in ["वा", "या", "सी", "की", "मी", "गर", "नी", "ती"] * 3))
    check(
        has(soup, "I:pseudo_script", language="mr"),
        "I+ non-Latin stem-recycling pseudo-script flagged",
    )
    check(
        has("the the the the the the the the the the the the the the the", "I:run")
        or has("the the the the the the the the the the the the the the the", "I:loop"),
        "I+ phrase loop flagged",
    )


FakeEngine, row = (T3.FakeEngine, T3.row)


def test_program():
    E = FakeEngine(
        lambda pre, a, n: (
            "1.\n2.\n"
            + [P1, P2, P3, P4][min(3, pre.count("***"))]
            + " ***extra tail words that should be cut away here"
        )
    )
    r = G.Program(E).run(
        row(
            "Write 3 paragraphs.", ("length_constraints:number_paragraphs", dict(num_paragraphs=3))
        ),
        None,
    )
    blocks = [x.strip() for x in r["text"].split("\n***\n")]
    check(
        blocks == [P1, P2, P3] and r["unmet"] == [],
        f"P-a. block hygiene strips bare numbers and cuts inline *** ({blocks!r:.120})",
    )
    check(
        r.get("program") == "gen4"
        and any((w.startswith("g4:") for w in r["weak"]))
        and any((l.startswith("g4:") for l in r["log"])),
        "P-a. guard firings recorded with 'g4:' prefix in log/weak, program='gen4'",
    )
    pre1 = "There are two different names that people call God, and here is the formatted version of the two responses:"
    E = FakeEngine(
        lambda pre, a, n: (
            (pre1 if a == 0 and "******" not in pre else P1 if "******" not in pre else P3)
            + "\n******\nignored"
        )
    )
    r = G.Program(E).run(row("Two responses.", ("combination:two_responses", {})), None)
    parts = [x.strip() for x in r["text"].split("******")]
    check(
        parts == [P1, P3] and "formatted version" not in r["text"],
        f"P-b. preamble block regenerated ({parts!r:.100})",
    )
    nat = "Here are some names: Liam, Noah, Oliver, Elijah and James are all popular choices for a baby boy this year."
    E = FakeEngine(lambda pre, a, n: "")
    r = G.Program(E).run(
        row(
            "10 names as bullets.", ("detectable_format:number_bullet_lists", dict(num_bullets=10))
        ),
        nat,
    )
    check(
        r["text"] == nat and "number_bullet_lists" in r["unmet"],
        f"P-c. no block built -> native answer kept, unmet={r['unmet']}",
    )
    concl = (
        "In conclusion, the plan is sound and the committee should adopt it without further delay."
    )
    nat = P1 + " " + P2 + " " + P4 + "\n\n" + concl
    E = FakeEngine(
        lambda pre, a, n: (
            P3
            + " "
            + "Members also discussed hiring two analysts to support the finance office next spring."
        )
    )
    r = G.Program(E).run(
        row(
            "At least 75 words.",
            ("length_constraints:number_words", dict(relation="at least", num_words=75)),
        ),
        nat,
    )
    check(
        r["text"].rstrip().endswith(concl)
        and P3 in r["text"]
        and (r["text"].index(P3) < r["text"].index(concl)),
        "P-d. continuation inserted before the conclusion",
    )
    check(
        E.calls and (not E.calls[0][0].endswith("delay. ")),
        "P-d. the continuation prompt ends before the conclusion",
    )
    E = FakeEngine(lambda pre, a, n: P1 + " " + P2)
    r = G.Program(E).run(
        row(
            "At least 80 words.",
            ("length_constraints:number_words", dict(relation="at least", num_words=80)),
        ),
        P1 + " " + P2 + " " + P3,
    )
    check(
        r["text"].count(P1) == 1 and "number_words" in r["unmet"],
        f"P-e. recycled continuation dropped, unmet={r['unmet']}",
    )
    nat = (
        "Raymond ruled Tripoli for many years.\n\n*Highlighted section Part 1*\n\n"
        + P1
        + "\n\n**Highlight** "
        + P2
        + "\n\n"
        + P3
    )
    E = FakeEngine(lambda pre, a, n: "unused")
    r = G.Program(E).run(
        row(
            "Highlight 2 sections.",
            ("detectable_format:number_highlighted_sections", dict(num_highlights=2)),
        ),
        nat,
    )
    hs = G.hl_official(r["text"])
    check(
        "Highlighted section" not in r["text"]
        and "Highlight**" not in r["text"]
        and (len(hs) == 2)
        and all((G.hl_valid4(r["text"], a, b, i) for a, b, i in hs)),
        f"P-f. meta highlights removed, 2 real highlights ({[h[2][:25] for h in hs]})",
    )
    ep = "Is there anything else I can help with?"
    nat = (
        'The sentiment could be positive or negative without more context.\n\nIn any case, the last sentence of my response will be "'
        + ep
        + '"'
    )
    r = G.Program(FakeEngine(lambda pre, a, n: "unused")).run(
        row("End with the phrase.", ("startend:end_checker", dict(end_phrase=ep))), nat
    )
    check(
        r["text"] == "The sentiment could be positive or negative without more context. " + ep,
        f"P-g. end-phrase meta fixed: {r['text'][-90:]!r}",
    )
    E = FakeEngine(lambda pre, a, n: "Hey sweetie, here are ten lovely names for a new baby boy:")
    r = G.Program(E).run(
        row("Add a P.P.S", ("detectable_content:postscript", dict(postscript_marker="P.P.S"))),
        P1 + " " + P2,
    )
    check(
        "P.P.S" not in r["text"] and "postscript" in r["unmet"],
        f"P-h. dangling P.S. rejected -> unmet ({r['unmet']})",
    )
    E = FakeEngine(lambda pre, a, n: "Remember to bring the signed forms on Monday morning.")
    r = G.Program(E).run(
        row("Add a P.S.", ("detectable_content:postscript", dict(postscript_marker="P.S."))),
        P1 + " " + P2 + " P.S.",
    )
    check(
        r["text"].endswith("P.S. Remember to bring the signed forms on Monday morning.")
        and r["text"].count("P.S.") == 1,
        f"P-h2. empty native P.S. replaced ({r['text'][-70:]!r})",
    )
    rq = "Write a haiku about moms."
    E = FakeEngine(
        lambda pre, a, n: [
            "Part #1: Repeat the request word for word.\n\nPart #2: Give your answer as a complete sentence.\n\nYour task is to write a haiku.",
            "A mother hums soft songs\nwhile the kettle starts to sing\nmorning light on bread",
        ][min(a, 1)]
    )
    r = G.Program(E).run(row("x", ("combination:repeat_prompt", dict(prompt_to_repeat=rq))), None)
    check(
        "kettle" in r["text"] and "Part #1" not in r["text"],
        f"P-h3. meta answer after the echo regenerated ({r['text'][-50:]!r})",
    )
    E = FakeEngine(
        lambda pre, a, n: (
            "The Quiet Committee\n"
            if pre.endswith("\n\nTitle: ") and len(pre) > 10
            else "title>>\n\nbody"
        )
    )
    r = G.Program(E).run(row("Title please.", ("detectable_format:title", {})), P1 + "\n\n" + P2)
    check(
        r["text"].startswith("<<The Quiet Committee>>\n\n" + P1) and r["unmet"] == [],
        f"P-i. body-conditioned title used ({r['text'][:40]!r})",
    )
    E = FakeEngine(
        lambda pre, a, n: [
            "asion] and [I just wanted to say hello to all of you today] ok.",
            "Your Name] will meet [Manager Name] at [Address].",
        ][min(a, 1)]
    )
    r = G.Program(E).run(
        row(
            "3 placeholders.", ("detectable_content:number_placeholders", dict(num_placeholders=3))
        ),
        P1,
    )
    phs = [m.group(1) for m in G.placeholders_in(r["text"])]
    check(
        phs == ["Your Name", "Manager Name", "Address"]
        and r["unmet"] == []
        and ("asion" not in r["text"]),
        f"P-j. fragment placeholders rejected ({phs})",
    )
    items = [
        "1 High-quality materials for long-lasting use 2 Sleek and minimalist design",
        "3 Compatibility with many modern devices",
    ]
    E = FakeEngine(
        lambda pre, a, n: (
            items[
                min(
                    1,
                    pre.count("\n* ") + (1 if pre.startswith("* ") and pre.count("* ") > 1 else 0),
                )
            ]
            + "\nmore"
        )
    )
    r = G.Program(E).run(
        row("2 bullets.", ("detectable_format:number_bullet_lists", dict(num_bullets=2))), None
    )
    check(
        r["text"]
        == "* High-quality materials for long-lasting use\n* Compatibility with many modern devices",
        f"P-k. bullet hygiene: {r['text']!r}",
    )
    bl = ["Fast charging keeps every device ready.", "Wireless pairing works with most phones."]
    E = FakeEngine(
        lambda pre, a, n: (
            bl[min(1, pre.count("* ") - 1)]
            if pre.endswith("* ")
            else "* Another bullet item here\n" + P3
        )
    )
    r = G.Program(E).run(
        row(
            "2 bullets, 40+ words.",
            ("detectable_format:number_bullet_lists", dict(num_bullets=2)),
            ("length_constraints:number_words", dict(relation="at least", num_words=30)),
        ),
        None,
    )
    check(
        r["text"].count("\n* ") + r["text"].startswith("* ") == 2 and "\n\n" + P3 in r["text"],
        f"P-l. continuation after bullets: {r['text']!r:.160}",
    )
    junk = "The navable freeways ar teh bestest trnsport fr goods an servces becuz thay allw botes to travl acros larje bodys of watr safly and quikly withot encountring problms or delys at al"
    E = FakeEngine(lambda pre, a, n: P1 + " " + P2)
    r = G.Program(E).run(row("No commas.", ("punctuation:no_comma", {})), junk)
    check(
        r["text"] == (P1 + " " + P2).replace(",", "")
        and any(("I:pseudo_words" in w for w in r["weak"])),
        f"P-m. pseudo-word native regenerated ({r['weak']})",
    )
    rid = "What has a dome but no doors? What am I?"
    E = FakeEngine(
        lambda pre, a, n: [
            "What am I?_ You and I I I Hi Hi I I I I I I I I",
            "You are the Capitol building, a landmark that has a dome and a vault and walls but never pays any rent.",
        ][min(a, 1)]
    )
    r = G.Program(E).run(row("x", ("combination:repeat_prompt", dict(prompt_to_repeat=rid))), None)
    check(
        r["text"].startswith(rid + "\n\n")
        and "Capitol" in r["text"]
        and (" I I I" not in r["text"]),
        f"P-n. echo soup regenerated: {r['text'][-60:]!r}",
    )
    nat = P1 + " " + P2
    r = G.Program(FakeEngine(lambda pre, a, n: "unused")).run(
        row(
            "Mention dose.",
            ("keywords:frequency", dict(keyword="dose", frequency=2, relation="at least")),
        ),
        nat,
    )
    check(
        G.needs_program(
            G.plan(
                row(
                    "x",
                    ("keywords:frequency", dict(keyword="dose", frequency=2, relation="at least")),
                )
            )
        )
        and r["text"] == nat,
        "P-o. guard-only prompt: native text kept for guarded certification",
    )


def test_guard():
    P = G.Program(FakeEngine(lambda pre, a, n: ""))

    def rej(orig, cert, **pl):
        base = dict(
            keywords=[],
            forbidden=[],
            placeholders=None,
            title=False,
            highlights=None,
            blocks=None,
            prefix="",
            postscript=None,
        )
        base.update(pl)
        return P.guard(orig, cert, base)[0]

    letter = "Dear Hiring Manager,\nI am writing to express my interest in the position. Please review my attached resume and I welcome the opportunity to discuss my application in person. Thank you for taking time to review my cover letter."
    cert = letter.replace("resume and I welcome", "resume << and I welcome").replace(
        "in person.", "in person >> today."
    )
    r = rej(letter, cert, title=True)
    check(
        any((x.startswith("g4:C:") for x in r)),
        f"G4-1. title wrapped around a mid-letter fragment rejected ({r})",
    )
    email = "Dear [Name],\nI just wanted to let you know that the boots were not quite perfect. Best regards,\n[Your Name]"
    cert = email.replace(
        "I just wanted to let you know that the boots",
        "[I just wanted to let you know that the boots]",
    ).replace("Best regards,", "[Best regards],")
    r = rej(email, cert, placeholders=4)
    check("g4:D:ph_fake" in r, f"G4-2. brackets around sentences rejected ({r})")
    steps = "Step one.\n* In year one it grows six inches.\n* In year two it grows again.\n* In year three it grows more.\nSo the answer is 8 feet tall."
    cert = steps.replace("* In year two", "***\n* In year two")
    r = rej(
        steps, cert, blocks=dict(kind="paragraphs", n=2, sep="\n***\n", block_prefix="", cut="\n\n")
    )
    check("g4:B:sep_mid_list" in r, f"G4-3. *** inserted mid-list rejected ({r})")
    art = "The peel protects the fruit. It is a complex organ made of layers.\n\nThe cells store water."
    cert = art.replace("layers.\n\n", "layers. disappointed================\n\n")
    r = rej(
        art,
        cert,
        keywords=["disappointed"],
        kw_req=[dict(kw="disappointed", relation="at least", n=1)],
    )
    check(
        any((x in r for x in ("g4:G:kw_fake", "g4:I:symbol_run"))),
        f"G4-4. keyword glued to junk separator rejected ({r})",
    )
    quiz = "Which kind of sofa is a chesterfield? It has a velvet cover and a backrest."
    cert = (
        quiz.replace("chesterfield", "cheserfield")
        .replace("velvet", "velvel")
        .replace("backrest", "backress")
    )
    r = rej(quiz, cert, letters=[dict(letter="t", relation="less than", n=1)])
    check(
        "g4:G:letter_mangled" in r and "g4:G:new_nonword" in r,
        f"G4-5. letter-avoiding mangled words rejected ({r})",
    )
    rid = "What has a dome but no doors? What am I?\n\n"
    r = rej(rid + "What am I?", rid + "What am I?_ You and I I I Hi Hi I I I I I I", prefix=rid)
    check(
        any((x.startswith("g4:I:") or x == "g4:H:echo_soup" for x in r)),
        f"G4-6. echo + token soup padding rejected ({r})",
    )
    ok = "The soldiers moved through the sector in the dark and the mission ended before dawn."
    check(not rej(ok, ok.replace("dark", "night sky")), "G4-7. a clean word edit passes")
    check(
        "g4:G:new_nonword" in rej(ok, ok.replace("moved", "orchestr moved")),
        "G4-8. a new non-word introduced by the MAP is rejected",
    )
    pr = "Raymond III, Count of Tripoli, was an important leader, and he ruled for years, as the sources say."
    r = rej(pr, pr.replace(", ", ". "))
    check(
        "g4:I:broken_sentences" in r,
        f"G4-9. commas turned into sentence-breaking periods rejected ({r})",
    )
    r = rej(P1 + "\n\n" + P2, P1 + "\n\n*Highlighted section Part 1*\n\n" + P2, highlights=1)
    check(
        any((x in r for x in ("g4:E:hl_fake", "junk_junk_highlight"))),
        f"G4-10. meta highlight introduced by the MAP rejected ({r})",
    )
    r = rej(
        P1 + " " + P2,
        P1 + " " + P2.replace("Friday", "Friday dose"),
        kw_req=[dict(kw="dose", relation="at least", n=1)],
        keywords=["dose"],
    )
    check(
        not any((x.startswith("g4:G:new") for x in r)),
        f"G4-11. inserting the keyword itself is not a non-word ({r})",
    )
    rq = "Write a haiku about moms with the keyword mom.\n\n"
    r = rej(
        rq + "Repeat the request.",
        rq + "mom. Repeat the request.",
        prefix=rq,
        keywords=["mom"],
        kw_req=[dict(kw="mom", relation="at least", n=1)],
    )
    check(
        "g4:G:kw_inserted_fake" in r,
        f"G4-16. keyword inserted outside a sentence while another fake already existed is rejected ({r})",
    )
    r = rej(
        "Dear [Name], I wanted to thank you for everything today.",
        "Dear [Name], [I wanted to thank you for everything today].",
        placeholders=3,
    )
    check(
        "g4:D:ph_inserted_fake" in r,
        f"G4-17. invalid placeholder added on top of an unmet count is rejected ({r})",
    )
    r = rej("Some steps follow.\n1 Mix\n2 Bake", "Some steps follow.\n1. Mix\n2. Bake.", min_sent=4)
    check(
        "g4:F:sent_inserted_fake" in r,
        f"G4-18. sentences padded by list-number periods rejected ({r})",
    )
    names = "Here are 10 possible names:\n\n* Aaron\n* Ethan\n* Grayson\n* Joshua\n* Lonzo"
    cert = "* are 10 possible names:\n\n* Aaron (* Ethan Eth* Grayson\n* Joshua and* Lon zo"
    r = rej(names, cert, blocks=dict(kind="bullets", n=3, sep="\n", block_prefix="* ", cut="\n"))
    check(
        "g4:A:stray_marker" in r and "g4:G:split_word" in r,
        f"G4-12. bullet markers spliced into a name list / split names rejected ({r})",
    )
    r = rej(
        "Skills:\n* Attention to detail.\n[Reference 1]",
        "Skills:\n**Attention to detail.\nSECTION 1]",
    )
    check(
        "g4:A:stray_marker" in r, f"G4-13. unbalanced markup introduced by the MAP rejected ({r})"
    )
    check(
        "g4:I:bad_chars" in rej(P1, P1.replace("budget", "bud�get")),
        "G4-14. replacement characters introduced by the MAP rejected",
    )
    check(
        not rej(
            "* Spelling matters.\n* Keep it short.",
            "* Spelling matters a lot.\n* Keep it short.",
            blocks=dict(kind="bullets", n=2, sep="\n", block_prefix="* ", cut="\n"),
        ),
        "G4-15. a clean edit inside bullets passes",
    )


if __name__ == "__main__":
    test_units()
    test_program()
    test_guard()
    raise SystemExit(bool(FAILS))
