"""
Alert and notification system
"""
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import cv2
from config.settings import get_settings

logger = logging.getLogger(__name__)

class AlertSystem:
    """Handle fire detection alerts and notifications"""
    
    def __init__(self, config=None):
        """Initialize alert system"""
        self.config = config or get_settings()
        self.alerts = []
        self.email_enabled = self.config.get('alerts.email_enabled', False)
        self.notification_enabled = self.config.get('alerts.notification_enabled', True)
        self.save_snapshot = self.config.get('alerts.save_snapshot', True)
        self.snapshot_dir = Path(self.config.get('alerts.snapshot_dir', './alerts/snapshots'))
        
        if self.save_snapshot:
            self.snapshot_dir.mkdir(parents=True, exist_ok=True)
    
    def trigger_alert(self, frame, fire_regions: List[dict], source: str = "unknown"):
        """
        Trigger alert when fire is detected
        
        Args:
            frame: Current frame
            fire_regions: List of detected fire regions
            source: Source identifier (camera, video file, etc.)
        """
        timestamp = datetime.now()
        alert_id = timestamp.strftime("%Y%m%d_%H%M%S_%f")
        
        alert_data = {
            'id': alert_id,
            'timestamp': timestamp,
            'source': source,
            'fire_regions': fire_regions,
            'region_count': len(fire_regions),
            'snapshot_path': None
        }
        
        # Save snapshot
        if self.save_snapshot and frame is not None:
            snapshot_path = self._save_snapshot(frame, alert_id)
            alert_data['snapshot_path'] = snapshot_path
        
        # Log alert
        logger.warning(f"FIRE ALERT! Detected {len(fire_regions)} fire region(s) "
                      f"from {source} at {timestamp}")
        
        # Send notifications
        if self.notification_enabled:
            self._send_notification(alert_data)
        
        if self.email_enabled:
            self._send_email_alert(alert_data)
        
        self.alerts.append(alert_data)
        return alert_data
    
    def _save_snapshot(self, frame, alert_id: str) -> Optional[str]:
        """Save snapshot of fire detection"""
        try:
            snapshot_filename = f"fire_alert_{alert_id}.jpg"
            snapshot_path = self.snapshot_dir / snapshot_filename
            
            cv2.imwrite(str(snapshot_path), frame)
            logger.info(f"Snapshot saved: {snapshot_path}")
            return str(snapshot_path)
        except Exception as e:
            logger.error(f"Failed to save snapshot: {e}")
            return None
    
    def _send_notification(self, alert_data: dict):
        """Send system notification"""
        try:
            import plyer
            plyer.notification.notify(
                title='FIRE DETECTED!',
                message=f"Fire detected in {alert_data['source']} - {len(alert_data['fire_regions'])} regions",
                timeout=10
            )
            logger.info("System notification sent")
        except Exception as e:
            logger.debug(f"Could not send system notification: {e}")
    
    def _send_email_alert(self, alert_data: dict):
        """Send email alert"""
        try:
            email_config = self.config.get_all().get('alerts', {})
            smtp_server = email_config.get('smtp_server', 'smtp.gmail.com')
            smtp_port = email_config.get('smtp_port', 587)
            sender_email = email_config.get('email_from')
            sender_password = email_config.get('email_password')
            recipients = email_config.get('email_to', [])
            
            if not all([sender_email, sender_password, recipients]):
                logger.warning("Email configuration incomplete")
                return
            
            msg = MIMEMultipart()
            msg['From'] = sender_email
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = f"🔥 FIRE ALERT - {alert_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}"
            
            body = f"""
            FIRE DETECTED!
            
            Source: {alert_data['source']}
            Time: {alert_data['timestamp']}
            Fire Regions Detected: {alert_data['region_count']}
            Alert ID: {alert_data['id']}
            Snapshot: {alert_data['snapshot_path']}
            
            Please take immediate action!
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(sender_email, sender_password)
                server.send_message(msg)
            
            logger.info(f"Email alert sent to {recipients}")
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
    
    def get_alerts(self, hours: int = 24) -> List[dict]:
        """Get recent alerts from last N hours"""
        from datetime import timedelta
        time_threshold = datetime.now() - timedelta(hours=hours)
        return [a for a in self.alerts if a['timestamp'] > time_threshold]
    
    def clear_old_snapshots(self, days: int = 7):
        """Delete old snapshot files"""
        from datetime import timedelta
        time_threshold = datetime.now() - timedelta(days=days)
        
        count = 0
        for snapshot in self.snapshot_dir.glob('*.jpg'):
            if datetime.fromtimestamp(snapshot.stat().st_mtime) < time_threshold:
                snapshot.unlink()
                count += 1
        
        logger.info(f"Deleted {count} old snapshot files")
    
    def get_alert_count(self) -> int:
        """Get total alert count"""
        return len(self.alerts)
