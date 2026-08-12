# s41 — measured metrics for the candidate remote tier (#659 cut 2 input)
# Taken 2026-08-12 from inside the serving container (192.168.0.101).

## Reachability
- eth0 = veth, reports 10000 Mb/s. Only route off-box: 192.168.0.0/24 via 192.168.0.2.
- 192.168.0.89 (hetero host "efeu-TC"): REACHABLE, key /root/.ssh/id_ed25519_192.168.0.89
  (needs -o IdentitiesOnly=yes).
- laptop efeu-TP14.fritz.box: UNREACHABLE (ping fails).
- Far side has enp1s0f0np0 @100000 Mb and enp1s0f1np1 @40000 Mb, BUT their
  subnets (169.254.17.33/16, 10.10.10.2/30, 192.168.40.10/24) are NOT routable
  from this container: 10.10.10.2 unreachable, 192.168.40.10 unreachable.
  169.254.17.33 answers ICMP but measures identically to the 1GbE address,
  i.e. it is being reached over the same 192.168.0.x path, not over the 100G NIC.
  => THE 40G/100G PATH IS NOT AVAILABLE TO THIS PROCESS.

## Measured latency
- RTT to 192.168.0.89: min/avg/max/mdev = 0.114/0.265/0.439/0.126 ms over 20 pkts,
  0% loss. (measured, provenance=measured)

## Measured bandwidth (dd 1500 MiB -> nc, wall clock)
- path A 192.168.0.89   : 75 MB/s
- path B 169.254.17.33  : 75 MB/s   (same path, confirms no fast NIC in play)
  => ~0.075 GB/s. Compare rig1.json's host:rig-2 claim of 2.83 GB/s "measured"
     over roce-40g: that number does NOT describe any link this process can use.

## Measured capacity of the candidate remote tier
- 192.168.0.89 total RAM 15277 MiB, available 8629 MiB, and it has a 64 GiB
  SWAPFILE (2.4 GiB already in use). Remote "RAM" is therefore swap-backed:
  it cannot honour a pinned-residency contract.
- One full-context kvso region is ~12.9 GB node-wide at ctx 393216 (C13,
  HANDOFF_677 2a). That does not fit in 8.6 GiB available, and at 0.075 GB/s a
  single region would take ~172 s to park.

## VERDICT
Rig-2 RAM is REFUSED for this rig by measurement, on TWO independent axes
(capacity insufficient, bandwidth ~38x below the local host tier). This is a
measured refusal, which is exactly what #407's Rate provenance model exists to
express -- it is a registry ENTRY, not a missing feature.
