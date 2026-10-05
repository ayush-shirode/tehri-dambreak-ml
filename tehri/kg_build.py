"""Step 1-2: source-traceable knowledge graph for the Tehri dam-break study.

Every numeric value is a dbk:Parameter node carrying value/min/max, unit, a status
(sourced | derived | assumption) and a dbk:source link to a dbk:Source node with
citation + URL. Scenario sampling (sampler.py) reads ranges ONLY from this graph.
"""
from rdflib import Graph, Namespace, Literal, RDF, RDFS, URIRef
from rdflib.namespace import XSD

DBK = Namespace("http://example.org/dambreak#")
g = Graph(); g.bind("dbk", DBK); g.bind("rdfs", RDFS)

SOURCES = {
    "THDC_Tehri": ("THDC India Ltd., Tehri Dam & HPP (Stage-I), technical parameters",
                   "https://thdc.co.in/en/projects/hydro/tehri-dam-HPP-stage-I"),
    "THDC_ICOLD2021": ("A. Singh, M. Mani, R. Vishnoi (THDC India Ltd.), 'Tehri Dam - saviour from extreme events', "
                       "ICOLD 2021 workshop on reservoir operation for handling extreme events, CBIP",
                       "https://www.cbip.org/ISRM-2022/ICOLD2021/Data/Workshop/Reservoir%20Operation%20for%20handling%20of%20Extreme%20Events/Tehri%20dam_Savior%20from%20extreme%20events_Final10-01-2020-17-31-07.pdf"),
    "Froehlich2008": ("D. C. Froehlich, J. Hydraul. Eng. 134(12), 1708-1721 (2008); equations as given in USACE HEC-RAS "
                      "Hydraulic Reference Manual, 'Estimating breach parameters'",
                      "https://www.hec.usace.army.mil/confluence/rasdocs/ras1dtechref/6.2/performing-a-dam-break-study-with-hec-ras/estimating-dam-breach-parameters/estimating-breach-parameters"),
    "VonThun1990": ("J. L. Von Thun and D. R. Gillette, 'Guidance on breach parameters', USBR internal memorandum (1990); "
                    "equations as given in USACE HEC-RAS Hydraulic Reference Manual",
                    "https://www.hec.usace.army.mil/confluence/rasdocs/ras1dtechref/6.2/performing-a-dam-break-study-with-hec-ras/estimating-dam-breach-parameters/estimating-breach-parameters"),
    "Froehlich1995": ("D. C. Froehlich, peak outflow regression (1995); equation as given in USACE HEC-RAS Hydraulic Reference Manual",
                      "https://www.hec.usace.army.mil/confluence/rasdocs/ras1dtechref/6.2/performing-a-dam-break-study-with-hec-ras/estimating-dam-breach-parameters/estimating-breach-parameters"),
    "Chow1959": ("V. T. Chow, Open-Channel Hydraulics (McGraw-Hill, 1959), Table 5-6; reproduced in FHWA HDS-6 Table 2.1",
                 "https://www.scdot.org/business/technicalPDFs/hydraulic/Chow%20from%20HDS%206.pdf"),
}
for k, (cit, url) in SOURCES.items():
    s = DBK[k]; g.add((s, RDF.type, DBK.Source)); g.add((s, DBK.citation, Literal(cit))); g.add((s, DBK.url, Literal(url, datatype=XSD.anyURI)))

def entity(name, cls, label):
    e = DBK[name]; g.add((e, RDF.type, DBK[cls])); g.add((e, RDFS.label, Literal(label))); return e

def param(owner, name, unit, status, src=None, value=None, vmin=None, vmax=None, note=None):
    p = DBK[f"{owner.split('#')[-1]}_{name}"]
    g.add((p, RDF.type, DBK.Parameter)); g.add((owner, DBK.hasParameter, p)); g.add((p, DBK.name, Literal(name)))
    g.add((p, DBK.unit, Literal(unit))); g.add((p, DBK.status, Literal(status)))
    for k, v in (("value", value), ("min", vmin), ("max", vmax)):
        if v is not None: g.add((p, DBK[k], Literal(float(v), datatype=XSD.double)))
    if src: g.add((p, DBK.source, DBK[src]))
    if note: g.add((p, RDFS.comment, Literal(note)))
    return p

# ---------------- physical entities ----------------
river = entity("Bhagirathi", "River", "Bhagirathi River")
dam = entity("TehriDam", "Dam", "Tehri Dam"); g.add((dam, DBK.onRiver, river))
res = entity("TehriReservoir", "Reservoir", "Tehri reservoir"); g.add((dam, DBK.impounds, res))
reach = entity("TehriDownstreamReach", "RiverReach", "Bhagirathi downstream of Tehri Dam"); g.add((dam, DBK.releasesInto, reach))
g.add((reach, DBK.partOf, river))

param(dam, "type", "-", "sourced", "THDC_Tehri", note="Earth and rockfill dam")
param(dam, "height", "m", "sourced", "THDC_Tehri", value=260.5)
param(dam, "crestLength", "m", "sourced", "THDC_Tehri", value=575)
param(dam, "crestElevation", "m a.s.l.", "sourced", "THDC_ICOLD2021", value=839.5)
param(dam, "spillwayCapacity", "m3/s", "sourced", "THDC_ICOLD2021", value=13043)
param(res, "FRL", "m a.s.l.", "sourced", "THDC_ICOLD2021", value=830)
param(res, "MWL", "m a.s.l.", "sourced", "THDC_ICOLD2021", value=835)
param(res, "MDDL", "m a.s.l.", "sourced", "THDC_ICOLD2021", value=740)
param(res, "grossStorage", "m3", "sourced", "THDC_Tehri", value=3540e6, note="gross storage at FRL")
param(res, "liveStorage", "m3", "sourced", "THDC_Tehri", value=2615e6)
param(res, "deadStorage", "m3", "derived", value=3540e6-2615e6, note="gross minus live storage = storage at MDDL")
param(res, "catchmentArea", "km2", "sourced", "THDC_ICOLD2021", value=7511)
param(res, "designFloodPMF", "m3/s", "sourced", "THDC_ICOLD2021", value=15540)
param(res, "maxObservedFlood", "m3/s", "sourced", "THDC_ICOLD2021", value=7500)

# ---------------- land cover / roughness ----------------
lc = entity("MountainStreamBoulders", "LandCoverClass", "Natural mountain stream, bed of cobbles with large boulders")
g.add((reach, DBK.hasLandCover, lc))
param(lc, "manningN", "s m^-1/3", "sourced", "Chow1959", vmin=0.040, vmax=0.070)
lc2 = entity("LightBrushTrees", "LandCoverClass", "Flood plain, light brush and trees (summer)")
g.add((reach, DBK.hasLandCover, lc2))
param(lc2, "manningN", "s m^-1/3", "sourced", "Chow1959", vmin=0.040, vmax=0.080)

# ---------------- breach models (equations + validity) ----------------
fr = entity("Froehlich2008Model", "BreachModel", "Froehlich (2008) regression")
g.add((fr, DBK.equation, Literal("B_avg = 0.27*K0*Vw^0.32*hb^0.04 [m]; t_f = 63.2*sqrt(Vw/(g*hb^2)) [s]")))
param(fr, "K0_overtopping", "-", "sourced", "Froehlich2008", value=1.3)
param(fr, "K0_piping", "-", "sourced", "Froehlich2008", value=1.0)
param(fr, "sideSlope_overtopping", "H:V", "sourced", "Froehlich2008", value=1.0)
param(fr, "sideSlope_piping", "H:V", "sourced", "Froehlich2008", value=0.7)
param(fr, "databaseDamHeight", "m", "sourced", "Froehlich2008", vmin=3.05, vmax=92.96)
param(fr, "databaseVolume", "m3", "sourced", "Froehlich2008", vmin=0.0139e6, vmax=660e6)
vt = entity("VonThun1990Model", "BreachModel", "Von Thun and Gillette (1990)")
g.add((vt, DBK.equation, Literal("t_f = 0.02*hw + 0.25 [h] (erosion resistant); t_f = 0.015*hw [h] (easily erodible)")))
param(vt, "databaseDamHeight", "m", "sourced", "VonThun1990", vmin=3.66, vmax=92.96)
fq = entity("Froehlich1995Peak", "PeakFlowModel", "Froehlich (1995) peak outflow")
g.add((fq, DBK.equation, Literal("Qp = 0.607*Vw^0.295*hw^1.24 [m3/s]")))
for m in (fr, vt): g.add((dam, DBK.breachModel, m))
g.add((dam, DBK.peakFlowCheck, fq))

# ---------------- scenario variables (what the sampler draws) ----------------
sc = entity("TehriScenarioSpace", "ScenarioSpace", "Uncertain inputs for Tehri breach scenarios")
param(sc, "reservoirLevel", "m a.s.l.", "sourced", "THDC_ICOLD2021", vmin=740, vmax=835,
      note="operating range MDDL..MWL; piping failures sampled in this range")
param(sc, "failureMode", "-", "sourced", "Froehlich2008", note="piping (K0=1.0, z=0.7) or overtopping (K0=1.3, z=1.0)")
param(sc, "breachTimeRange", "s", "derived", note="per scenario: between Froehlich (2008) and Von Thun-Gillette (1990, erosion "
      "resistant) estimates; both equations are extrapolated because Tehri exceeds their database range")
param(sc, "breachWidthFactor", "-", "assumption", vmin=0.7, vmax=1.3,
      note="ASSUMPTION: +/-30% about Froehlich (2008) width to represent extrapolation uncertainty; no sourced interval")
param(sc, "manningN", "s m^-1/3", "sourced", "Chow1959", vmin=0.040, vmax=0.070)

g.serialize("tehri_kg.ttl", format="turtle")
print("triples:", len(g))
