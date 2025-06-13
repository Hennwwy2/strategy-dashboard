# ai_debug_system.py - Advanced AI Debugging System
import streamlit as st
import anthropic
import traceback
import sys
import logging
import json
import sqlite3
from datetime import datetime
import pandas as pd
from contextlib import contextmanager
import inspect
import os

class AIDebugSystem:
    def __init__(self, claude_api_key=None):
        self.claude_api_key = claude_api_key or self._get_claude_key()
        self.client = None
        if self.claude_api_key and self.claude_api_key != "your-claude-api-key":
            try:
                self.client = anthropic.Anthropic(api_key=self.claude_api_key)
            except:
                st.warning("Claude API not available - debugging will work in offline mode")
        
        self.debug_db = "debug_logs.db"
        self.setup_debug_database()
        self.setup_logging()
    
    def _get_claude_key(self):
        """Get Claude API key from secrets or config"""
        try:
            return st.secrets.get("claude_api_key", "your-claude-api-key")
        except:
            return "your-claude-api-key"
    
    def setup_debug_database(self):
        """Setup SQLite database for debug logs"""
        conn = sqlite3.connect(self.debug_db)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS debug_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                error_type TEXT,
                error_message TEXT,
                traceback_info TEXT,
                function_name TEXT,
                context_data TEXT,
                ai_diagnosis TEXT,
                resolution_status TEXT DEFAULT 'open',
                user_notes TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS debug_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_start TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                session_data TEXT,
                ai_conversation TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def setup_logging(self):
        """Setup comprehensive logging"""
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('debug.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('TradingDashboard')
    
    def capture_system_context(self):
        """Capture current system state for debugging"""
        context = {
            "timestamp": datetime.now().isoformat(),
            "streamlit_session": dict(st.session_state) if hasattr(st, 'session_state') else {},
            "python_version": sys.version,
            "working_directory": os.getcwd(),
            "environment_vars": {k: v for k, v in os.environ.items() if 'KEY' not in k and 'SECRET' not in k},
        }
        
        # Add trading-specific context
        try:
            if hasattr(st.session_state, 'paper_db'):
                # Get recent positions and trades
                conn = sqlite3.connect('paper_trading.db')
                context["recent_trades"] = pd.read_sql_query(
                    "SELECT * FROM trades ORDER BY timestamp DESC LIMIT 5", conn
                ).to_dict('records')
                context["current_positions"] = pd.read_sql_query(
                    "SELECT * FROM positions WHERE quantity != 0", conn
                ).to_dict('records')
                conn.close()
        except:
            context["trading_context"] = "Unable to capture trading context"
        
        return context
    
    def ai_analyze_error(self, error_info, context=None):
        """Get AI analysis of error"""
        if not self.client:
            return "Claude API not available. Please add your Claude API key to secrets."
        
        context_str = json.dumps(context, indent=2) if context else "No additional context"
        
        prompt = f"""
        You are an expert Python/Streamlit developer debugging a trading dashboard application.
        
        SYSTEM CONTEXT:
        - Application: Streamlit-based trading dashboard
        - Data Sources: Polygon.io (market data), Tiingo (historical data)
        - Database: SQLite for paper trading
        - Key Libraries: streamlit, pandas, requests, plotly, matplotlib
        
        ERROR DETAILS:
        Type: {error_info.get('type', 'Unknown')}
        Message: {error_info.get('message', 'No message')}
        Function: {error_info.get('function', 'Unknown')}
        
        TRACEBACK:
        {error_info.get('traceback', 'No traceback available')}
        
        SYSTEM CONTEXT:
        {context_str}
        
        Please provide:
        1. **Root Cause Analysis**: What exactly caused this error?
        2. **Immediate Fix**: Code changes needed to resolve this specific error
        3. **Prevention Strategy**: How to prevent similar errors in the future
        4. **Testing Approach**: How to verify the fix works
        5. **Monitoring**: What to watch for to catch similar issues early
        
        Format your response in clear sections with code examples where helpful.
        """
        
        try:
            response = self.client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=2000,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            return response.content[0].text if hasattr(response.content[0], 'text') else str(response.content[0])
        except Exception as e:
            return f"AI Analysis failed: {str(e)}"
    
    def log_error(self, error_info, context=None, ai_diagnosis=None):
        """Log error to database"""
        conn = sqlite3.connect(self.debug_db)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO debug_logs 
            (error_type, error_message, traceback_info, function_name, context_data, ai_diagnosis)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            error_info.get('type', ''),
            error_info.get('message', ''),
            error_info.get('traceback', ''),
            error_info.get('function', ''),
            json.dumps(context) if context else '',
            ai_diagnosis or ''
        ))
        
        conn.commit()
        conn.close()
    
    def smart_exception_handler(self, exc_type, exc_value, exc_traceback):
        """Advanced exception handler with AI analysis"""
        error_info = {
            "type": exc_type.__name__,
            "message": str(exc_value),
            "traceback": ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback)),
            "function": self._get_function_from_traceback(exc_traceback)
        }
        
        # Capture system context
        context = self.capture_system_context()
        
        # Get AI analysis
        ai_diagnosis = self.ai_analyze_error(error_info, context)
        
        # Log everything
        self.log_error(error_info, context, ai_diagnosis)
        
        # Display in Streamlit
        self._display_error_analysis(error_info, ai_diagnosis)
    
    def _get_function_from_traceback(self, tb):
        """Extract function name from traceback"""
        try:
            while tb.tb_next:
                tb = tb.tb_next
            return tb.tb_frame.f_code.co_name
        except:
            return "unknown"
    
    def _display_error_analysis(self, error_info, ai_diagnosis):
        """Display error analysis in Streamlit"""
        st.error(f"🐛 **Error Detected**: {error_info['type']}")
        st.write(f"**Message**: {error_info['message']}")
        
        with st.expander("🤖 AI Diagnosis & Solution", expanded=True):
            if ai_diagnosis:
                st.markdown(ai_diagnosis)
            else:
                st.warning("AI analysis not available")
        
        with st.expander("🔍 Technical Details"):
            st.code(error_info['traceback'], language='python')
    
    @contextmanager
    def debug_function(self, func_name=None):
        """Context manager for debugging specific functions"""
        func_name = func_name or inspect.stack()[1].function
        start_time = datetime.now()
        
        self.logger.info(f"Starting function: {func_name}")
        
        try:
            yield
            self.logger.info(f"Completed function: {func_name} in {datetime.now() - start_time}")
        except Exception as e:
            error_info = {
                "type": type(e).__name__,
                "message": str(e),
                "traceback": traceback.format_exc(),
                "function": func_name
            }
            
            context = self.capture_system_context()
            ai_diagnosis = self.ai_analyze_error(error_info, context)
            self.log_error(error_info, context, ai_diagnosis)
            
            # Display in Streamlit
            self._display_error_analysis(error_info, ai_diagnosis)
            raise
    
    def debug_decorator(self, func):
        """Decorator for automatic function debugging"""
        def wrapper(*args, **kwargs):
            with self.debug_function(func.__name__):
                return func(*args, **kwargs)
        return wrapper

def create_debug_panel():
    """Create the development-focused debug panel"""
    st.sidebar.markdown("---")
    st.sidebar.subheader("🛠️ Debug Assistant")
    
    debug_mode = st.sidebar.selectbox(
        "Debug Mode:",
        ["Manual Debug", "Error Logs", "Live Monitoring", "Code Analysis"]
    )
    
    if debug_mode == "Manual Debug":
        manual_debug_panel()
    elif debug_mode == "Error Logs":
        error_logs_panel()
    elif debug_mode == "Live Monitoring":
        live_monitoring_panel()
    elif debug_mode == "Code Analysis":
        code_analysis_panel()

def manual_debug_panel():
    """Manual debugging interface"""
    st.sidebar.write("**Manual Debugging**")
    
    error_input = st.sidebar.text_area("Paste error message:", height=100)
    code_context = st.sidebar.text_area("Code context (optional):", height=80)
    
    if st.sidebar.button("🤖 Get AI Analysis"):
        if error_input:
            debug_system = AIDebugSystem()
            
            error_info = {
                "type": "Manual Entry",
                "message": error_input,
                "traceback": error_input,
                "function": "manual_debug"
            }
            
            context = {"code_context": code_context} if code_context else None
            diagnosis = debug_system.ai_analyze_error(error_info, context)
            
            st.write("### 🤖 AI Diagnosis")
            st.markdown(diagnosis)
        else:
            st.sidebar.error("Please enter an error message")

def error_logs_panel():
    """Display error logs from database"""
    try:
        conn = sqlite3.connect("debug_logs.db")
        logs_df = pd.read_sql_query(
            "SELECT * FROM debug_logs ORDER BY timestamp DESC LIMIT 20", conn
        )
        conn.close()
        
        if not logs_df.empty:
            st.sidebar.write(f"**Recent Errors ({len(logs_df)})**")
            
            selected_log = st.sidebar.selectbox(
                "Select error to view:",
                options=range(len(logs_df)),
                format_func=lambda x: f"{logs_df.iloc[x]['timestamp'][:16]} - {logs_df.iloc[x]['error_type']}"
            )
            
            if st.sidebar.button("View Details"):
                log_details = logs_df.iloc[selected_log]
                
                st.write("### 🐛 Error Details")
                st.write(f"**Time**: {log_details['timestamp']}")
                st.write(f"**Type**: {log_details['error_type']}")
                st.write(f"**Function**: {log_details['function_name']}")
                
                with st.expander("Error Message & Traceback"):
                    st.code(log_details['traceback_info'])
                
                if log_details['ai_diagnosis']:
                    with st.expander("🤖 AI Diagnosis", expanded=True):
                        st.markdown(log_details['ai_diagnosis'])
        else:
            st.sidebar.info("No error logs found")
    except:
        st.sidebar.warning("Error logs database not available")

def live_monitoring_panel():
    """Live system monitoring"""
    st.sidebar.write("**Live Monitoring**")
    
    if st.sidebar.button("📊 System Health Check"):
        st.write("### 📊 System Health Report")
        
        # Check database connections
        try:
            conn = sqlite3.connect("paper_trading.db")
            conn.close()
            st.success("✅ Paper trading database: Connected")
        except:
            st.error("❌ Paper trading database: Connection failed")
        
        # Check API endpoints
        try:
            import requests
            response = requests.get("https://api.polygon.io/v2/aggs/ticker/AAPL/range/1/day/2023-01-01/2023-01-02?apikey=test")
            if response.status_code in [200, 401]:  # 401 means endpoint works but auth failed
                st.success("✅ Polygon API: Endpoint reachable")
            else:
                st.warning("⚠️ Polygon API: Unexpected response")
        except:
            st.error("❌ Polygon API: Connection failed")
        
        # Memory usage
        try:
            import psutil
            memory = psutil.virtual_memory()
            st.info(f"💾 Memory Usage: {memory.percent}%")
        except:
            st.info("💾 Memory monitoring not available")

def code_analysis_panel():
    """Code analysis and suggestions"""
    st.sidebar.write("**Code Analysis**")
    
    code_input = st.sidebar.text_area("Paste code for analysis:", height=150)
    analysis_type = st.sidebar.radio(
        "Analysis type:",
        ["Bug Check", "Performance Review", "Best Practices", "Security Scan"]
    )
    
    if st.sidebar.button("🔍 Analyze Code"):
        if code_input:
            debug_system = AIDebugSystem()
            
            # Create analysis prompt based on type
            prompts = {
                "Bug Check": f"Review this code for potential bugs and errors:\n\n{code_input}",
                "Performance Review": f"Analyze this code for performance issues and optimization opportunities:\n\n{code_input}",
                "Best Practices": f"Review this code for Python/Streamlit best practices:\n\n{code_input}",
                "Security Scan": f"Check this code for security vulnerabilities:\n\n{code_input}"
            }
            
            if debug_system.client:
                try:
                    response = debug_system.client.messages.create(
                        model="claude-3-sonnet-20240229",
                        max_tokens=1500,
                        messages=[{
                            "role": "user",
                            "content": prompts[analysis_type]
                        }]
                    )
                    
                    st.write(f"### 🔍 {analysis_type} Results")
                    analysis_result = response.content[0].text if hasattr(response.content[0], 'text') else str(response.content[0])
                    st.markdown(analysis_result)
                except Exception as e:
                    st.error(f"Analysis failed: {e}")
            else:
                st.error("Claude API not available for code analysis")

# Integration functions for the main dashboard
def integrate_ai_debugging():
    """Initialize AI debugging system for the dashboard"""
    if 'debug_system' not in st.session_state:
        st.session_state.debug_system = AIDebugSystem()
    
    # Set up automatic exception handling
    sys.excepthook = st.session_state.debug_system.smart_exception_handler
    
    return st.session_state.debug_system

def debug_wrapper(func):
    """Decorator to wrap functions with debugging"""
    def wrapper(*args, **kwargs):
        debug_system = st.session_state.get('debug_system')
        if debug_system:
            with debug_system.debug_function(func.__name__):
                return func(*args, **kwargs)
        else:
            return func(*args, **kwargs)
    return wrapper

# Usage example for your trading functions
@debug_wrapper
def example_trading_function():
    """Example of how to use the debug wrapper"""
    # Your trading logic here
    pass