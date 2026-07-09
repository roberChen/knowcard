package search

// FusedResult is a single result after RRF fusion of multiple retrieval lanes.
type FusedResult struct {
	DocID    string
	Score    float64
	HitTypes []string // which lanes matched: "semantic", "keyword"
}

// RRF (Reciprocal Rank Fusion) combines ranked result lists from multiple
// retrieval methods into a single ranking.
// score(doc) = sum over all lanes: 1 / (k + rank_in_lane(doc))
// where rank starts at 1. k is a tuning constant (typically 60).
func RRF(lanes [][]SearchResult, k int) []FusedResult {
	if k <= 0 {
		k = 60
	}

	type laneName struct{}

	// Map: docID -> accumulated score, and which lanes hit it
	type docInfo struct {
		score    float64
		hitTypes []string
	}
	infos := make(map[string]*docInfo)

	for laneIdx, lane := range lanes {
		var laneName string
		switch laneIdx {
		case 0:
			laneName = "semantic"
		case 1:
			laneName = "keyword"
		default:
			laneName = "lane"
		}

		for rank, res := range lane {
			if _, ok := infos[res.DocID]; !ok {
				infos[res.DocID] = &docInfo{}
			}
			infos[res.DocID].score += 1.0 / float64(k+rank+1)
			infos[res.DocID].hitTypes = append(infos[res.DocID].hitTypes, laneName)
		}
	}

	// Convert to sorted slice
	results := make([]FusedResult, 0, len(infos))
	for docID, info := range infos {
		hitType := "both"
		if len(info.hitTypes) == 1 {
			hitType = info.hitTypes[0]
		}
		results = append(results, FusedResult{
			DocID:    docID,
			Score:    info.score,
			HitTypes: []string{hitType},
		})
	}

	// Sort by score descending
	for i := 0; i < len(results); i++ {
		for j := i + 1; j < len(results); j++ {
			if results[j].Score > results[i].Score {
				results[i], results[j] = results[j], results[i]
			}
		}
	}

	return results
}

// RRFWithNames is like RRF but allows naming each lane explicitly.
func RRFWithNames(lanes []struct {
	Name    string
	Results []SearchResult
}, k int) []FusedResult {
	if k <= 0 {
		k = 60
	}

	type docInfo struct {
		score    float64
		hitTypes []string
	}
	infos := make(map[string]*docInfo)

	for _, lane := range lanes {
		for rank, res := range lane.Results {
			if _, ok := infos[res.DocID]; !ok {
				infos[res.DocID] = &docInfo{}
			}
			infos[res.DocID].score += 1.0 / float64(k+rank+1)
			infos[res.DocID].hitTypes = append(infos[res.DocID].hitTypes, lane.Name)
		}
	}

	results := make([]FusedResult, 0, len(infos))
	for docID, info := range infos {
		hitType := "both"
		if len(info.hitTypes) == 1 {
			hitType = info.hitTypes[0]
		}
		results = append(results, FusedResult{
			DocID:    docID,
			Score:    info.score,
			HitTypes: []string{hitType},
		})
	}

	// Sort by score descending
	for i := 0; i < len(results); i++ {
		for j := i + 1; j < len(results); j++ {
			if results[j].Score > results[i].Score {
				results[i], results[j] = results[j], results[i]
			}
		}
	}

	return results
}
