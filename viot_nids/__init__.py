"""V5: unsupervised one-second directional Video-IoT flow detector."""

FEATURE_NAMES = (
    "packet_count", "byte_count", "active_duration",
    "packet_len_mean", "packet_len_std", "packet_len_min", "packet_len_max",
    "packet_len_p25", "packet_len_p50", "packet_len_p75",
    "iat_mean", "iat_std", "iat_min", "iat_max", "iat_p25", "iat_p50",
    "iat_p75", "iat_cv", "active_bin_count", "bin_bytes_mean",
    "bin_bytes_std", "bin_bytes_max", "bin_packets_mean",
    "bin_packets_std", "bin_packets_max",
)
