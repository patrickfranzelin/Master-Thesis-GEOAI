
#  Bewertungsschema für Gebäudegeometrien

| Kategorie   | Definition                                                     | Wann wählen?                                                                                                              | Beispiele                                               |
| ----------- | -------------------------------------------------------------- |---------------------------------------------------------------------------------------------------------------------------| ------------------------------------------------------- |
| **Perfect** | Geometrie entspricht vollständig der tatsächlichen Gebäudeform | - Keine sichtbaren Fehler<br>- Kanten sauber und korrekt<br>- Lage korrekt                                                | Rechteckiges Haus exakt getroffen                       |
| **Good**    | Kleine Fehler vorhanden, aber Gesamtform korrekt               | - Leichte Ungenauigkeiten<br>- Kleine Versätze oder Rundungen<br>- Struktur stimmt insgesamt                              | Minimal verschoben, leicht ungenaue Kanten              |
| **OK**      | Deutliche Fehler, aber Gebäude noch erkennbar                  | - Mehrere Fehler<br>- Teile fehlen oder sind falsch<br>- Form teilweise verzerrt, Lage falsch aber ungef#hr richtige form | Dach fehlt teilweise, unregelmäßige Form                |
| **Bad**     | Geometrie stark fehlerhaft oder unbrauchbar                    | - Große Teile fehlen<br>- falsche Lage und falsche form 3<br>- komplett falsche Form                                      | Gebäude stark verschoben oder völlig falsch segmentiert |

---

# Fehlerkategorien (für Mehrfachauswahl)

| Fehler             | Definition                            | Wann wählen?                        |
| ------------------ | ------------------------------------- | ----------------------------------- |
| **SHAPE_MISMATCH** | Form stimmt nicht mit Gebäude überein | Falsche Winkel, verzerrte Konturen  |
| **MISSING_PARTS**  | Teile des Gebäudes fehlen             | L-Form wird zu Rechteck             |
| **EXTRA_PARTS**    | Zusätzliche Geometrie enthalten       | Garten / Straße mit drin            |
| **OVERSIMPLIFIED** | Geometrie zu stark vereinfacht        | Komplexe Struktur wird glattgezogen |
| **SHIFTED**        | Gebäude ist räumlich verschoben       | Form stimmt, aber falsche Position  |

---

#  Post vs SAM Bewertung

| Kategorie | Bedeutung                                        |
| --------- | ------------------------------------------------ |
| **No**    | Postprocessing hat keine neuen Fehler eingeführt |
| **Yes**   | Postprocessing hat zusätzliche Fehler erzeugt    |

---
