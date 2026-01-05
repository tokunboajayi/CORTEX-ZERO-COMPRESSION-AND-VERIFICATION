from sqlalchemy import create_engine, Column, String, Float, Integer, JSON
from sqlalchemy.orm import declarative_base, sessionmaker
from typing import List, Optional
import json

from .models import AtomicClaim, Opinion, Evidence

Base = declarative_base()

class DBClaim(Base):
    __tablename__ = 'claims'
    
    id = Column(String, primary_key=True)
    subject = Column(String)
    predicate = Column(String)
    object = Column(String)
    creation_time = Column(Integer)
    current_ess = Column(Float)
    
    # Store complex objects as JSON
    opinion_json = Column(JSON) # belief, disbelief, uncertainty
    history_json = Column(JSON) # List of (t, ess)
    
class DBEvidence(Base):
    __tablename__ = 'evidence'
    
    id = Column(String, primary_key=True)
    claim_id = Column(String)
    content = Column(String)
    source_id = Column(String)
    timestamp = Column(Integer)
    polarity = Column(String)

class DBChatSession(Base):
    __tablename__ = 'chat_sessions'
    id = Column(String, primary_key=True)
    title = Column(String)
    created_at = Column(Integer)

class DBChatMessage(Base):
    __tablename__ = 'chat_messages'
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String)
    role = Column(String) # 'user' or 'bot'
    content = Column(String)
    evidence_json = Column(JSON) # Store truth vectors cited
    timestamp = Column(Integer)

class DBKnowledgeGap(Base):
    __tablename__ = 'knowledge_gaps'
    id = Column(Integer, primary_key=True, autoincrement=True)
    query = Column(String)
    status = Column(String, default="OPEN") # OPEN, RESOLVED
    timestamp = Column(Integer)

class PersistenceManager:
    def __init__(self, db_url=None):
        if db_url is None:
            import os
            base_dir = os.path.dirname(os.path.abspath(__file__))
            # Go up one level to project root if being called from within a package
            if base_dir.endswith('cortex_zero_core'):
                 base_dir = os.path.dirname(base_dir)
            db_path = os.path.join(base_dir, "cortex_zero.db")
            db_url = f"sqlite:///{db_path}"
        
        print(f"DEBUG: Connecting to database at {db_url}")
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        
    def save_claim(self, claim: AtomicClaim):
        session = self.Session()
        # Upsert logic (simple deletion+insert or merge)
        # For MVP, merge is better
        
        # Merge is tricky with default declarative. Let's try to get existing, update or create.
        existing = session.query(DBClaim).filter_by(id=claim.id).first()
        
        op_dict = {
            "belief": claim.current_opinion.belief,
            "disbelief": claim.current_opinion.disbelief,
            "uncertainty": claim.current_opinion.uncertainty,
            "base_rate": claim.current_opinion.base_rate
        }
        
        if existing:
            existing.current_ess = claim.current_ess
            existing.opinion_json = op_dict
            existing.history_json = claim.history
        else:
            new_db_claim = DBClaim(
                id=claim.id,
                subject=claim.subject,
                predicate=claim.predicate,
                object=claim.object,
                creation_time=claim.creation_time,
                current_ess=claim.current_ess,
                opinion_json=op_dict,
                history_json=claim.history
            )
            session.add(new_db_claim)
            
        session.commit()
        session.close()

    def bulk_save_claims(self, claims: List[AtomicClaim]):
        session = self.Session()
        print(f"DEBUG: Bulk saving {len(claims)} claims...")
        for claim in claims:
            op_dict = {
                "belief": claim.current_opinion.belief,
                "disbelief": claim.current_opinion.disbelief,
                "uncertainty": claim.current_opinion.uncertainty,
                "base_rate": claim.current_opinion.base_rate
            }
            new_db_claim = DBClaim(
                id=claim.id,
                subject=claim.subject,
                predicate=claim.predicate,
                object=claim.object,
                creation_time=claim.creation_time,
                current_ess=claim.current_ess,
                opinion_json=op_dict,
                history_json=claim.history
            )
            session.merge(new_db_claim)
        session.commit()
        session.close()

    def search_claims(self, keywords: List[str], limit=5) -> List[AtomicClaim]:
        session = self.Session()
        from sqlalchemy import or_
        filters = []
        for w in keywords:
            filters.append(DBClaim.subject.ilike(f"%{w}%"))
            filters.append(DBClaim.object.ilike(f"%{w}%"))
            filters.append(DBClaim.predicate.ilike(f"%{w}%"))
        
        # Fetch a broader set of candidates
        db_results = session.query(DBClaim).filter(or_(*filters)).limit(50).all()
        
        # In-Memory Scoring
        scored = []
        garbage_subjects = {"he", "she", "it", "they", "this", "that", "who", "which"}
        
        for row in db_results:
            if row.subject.lower() in garbage_subjects: continue

            score = 0
            # 1. ESS is base score (0-1)
            score += row.current_ess
            
            # 2. Keyword Overlap
            matches = 0
            txt = f"{row.subject} {row.predicate} {row.object}".lower()
            for k in keywords:
                if k in txt: 
                    matches += 1
                    # TIER 1: EXACT SUBJECT MATCH (The "Definition" Bonus)
                    if row.subject.lower() == k:
                        score += 50
                    # TIER 2: SUBJECT CONTAINS KEYWORD
                    elif k in row.subject.lower():
                        score += 20
                    # TIER 3: OBJECT CONTAINS KEYWORD
                    elif k in row.object.lower():
                        score += 5
            
            # 3. Predicate Bonus (Prefer definitions)
            if row.predicate.lower().strip() in ["is", "was", "are", "refers to", "means", "is_a"]:
                score += 25
                
            # 4. Conciseness Bonus (Prefer short facts over long stories)
            if len(row.object) < 100:
                score += 5
            
            # Bonus for complete coverage
            if matches >= 2: score += 10
            
            scored.append((score, row))
            
        # Sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)
        
        result = []
        for _, row in scored[:limit]:
            op_data = row.opinion_json
            op = Opinion(
                belief=op_data['belief'],
                disbelief=op_data['disbelief'],
                uncertainty=op_data['uncertainty'],
                base_rate=op_data.get('base_rate', 0.5)
            )
            claim = AtomicClaim(
                id=row.id,
                subject=row.subject,
                predicate=row.predicate,
                object=row.object,
                creation_time=row.creation_time,
                current_ess=row.current_ess,
                history=[tuple(x) for x in row.history_json],
                current_opinion=op
            )
            result.append(claim)
        session.close()
        return result

    def get_all_claims(self) -> List[AtomicClaim]:
        session = self.Session()
        db_claims = session.query(DBClaim).all()
        result = []
        skipped = 0
        
        for row in db_claims:
            try:
                # Skip records with None values (corrupted data)
                if not row.subject or not row.predicate or not row.object or row.creation_time is None:
                    skipped += 1
                    continue
                    
                op_data = row.opinion_json
                op = Opinion(
                    belief=op_data['belief'],
                    disbelief=op_data['disbelief'],
                    uncertainty=op_data['uncertainty'],
                    base_rate=op_data.get('base_rate', 0.5)
                )
                claim = AtomicClaim(
                    id=row.id,
                    subject=row.subject,
                    predicate=row.predicate,
                    object=row.object,
                    creation_time=row.creation_time,
                    current_ess=row.current_ess,
                    history=[tuple(x) for x in row.history_json] if row.history_json else [],
                    current_opinion=op
                )
                result.append(claim)
            except Exception as e:
                print(f"WARNING: Skipping corrupt claim {row.id}: {e}")
                skipped += 1
                continue
                
        if skipped > 0:
            print(f"INFO: Skipped {skipped} corrupt database records")
        session.close()
        return result
    
    def save_evidence(self, ev: Evidence, claim_id: str):
        session = self.Session()
        # Check exist
        if session.query(DBEvidence).filter_by(id=ev.id).first():
            session.close()
            return

        db_ev = DBEvidence(
            id=ev.id,
            claim_id=claim_id,
            content=ev.content,
            source_id=ev.source_id,
            timestamp=ev.timestamp,
            polarity=ev.polarity.value
        )
        session.add(db_ev)
        session.commit()
        session.close()

    # --- Chat Persistence Methods ---
    def create_chat_session(self, session_id: str, title: str):
        session = self.Session()
        import time
        # Check if exists
        if session.query(DBChatSession).filter_by(id=session_id).first():
            session.close()
            return
        db_sess = DBChatSession(id=session_id, title=title, created_at=int(time.time()))
        session.add(db_sess)
        session.commit()
        session.close()

    def add_chat_message(self, session_id: str, role: str, content: str, evidence: Optional[dict] = None):
        session = self.Session()
        import time
        msg = DBChatMessage(
            session_id=session_id,
            role=role,
            content=content,
            evidence_json=evidence,
            timestamp=int(time.time())
        )
        session.add(msg)
        session.commit()
        session.close()

    def get_chat_sessions(self):
        session = self.Session()
        results = session.query(DBChatSession).order_by(DBChatSession.created_at.desc()).all()
        out = [{"id": s.id, "title": s.title} for s in results]
        session.close()
        return out

    def get_chat_messages(self, session_id: str):
        session = self.Session()
        results = session.query(DBChatMessage).filter_by(session_id=session_id).order_by(DBChatMessage.timestamp.asc()).all()
        out = [{
            "role": m.role,
            "content": m.content,
            "evidence": m.evidence_json,
            "timestamp": m.timestamp
        } for m in results]
        session.close()
        return out

    def log_knowledge_gap(self, query: str):
        session = self.Session()
        import time
        # Dedup: check if open gap exists for this query
        exists = session.query(DBKnowledgeGap).filter_by(query=query, status="OPEN").first()
        if not exists:
            gap = DBKnowledgeGap(query=query, timestamp=int(time.time()))
            session.add(gap)
            session.commit()
        session.close()

    def get_open_gaps(self) -> List[dict]:
        session = self.Session()
        gaps = session.query(DBKnowledgeGap).filter_by(status="OPEN").all()
        out = [{"id": g.id, "query": g.query} for g in gaps]
        session.close()
        return out
    
    def resolve_gap(self, gap_id: int):
        session = self.Session()
        gap = session.query(DBKnowledgeGap).filter_by(id=gap_id).first()
        if gap:
            gap.status = "RESOLVED"
            session.commit()
        session.close()
