"""Journal a temporary automatic-reboot hold without disabling package updates."""
import argparse,json,os,socket,subprocess
from datetime import datetime,timezone
from pathlib import Path

CONFIG=Path('/etc/apt/apt.conf.d/99zz-zen-research-reboot-hold')
SCHEDULED=Path('/run/systemd/shutdown/scheduled')
CONTENT='// Preserve active research jobs; remove after goal completion.\nUnattended-Upgrade::Automatic-Reboot "false";\n'
HOSTS=('r3500','r3800x','r5600g','i134','r7900x','i270')


def settings():
    return subprocess.check_output(['apt-config','shell','ZenResearchAutoReboot',
        'Unattended-Upgrade::Automatic-Reboot','ZenResearchRebootTime',
        'Unattended-Upgrade::Automatic-Reboot-Time'],text=True)


def hold(output):
    host=socket.gethostname()
    if host not in HOSTS or os.geteuid()!=0:raise ValueError('approved worker and root required')
    if CONFIG.exists() or output.exists():raise ValueError('preserve existing hold or operation journal')
    before=settings()
    if "ZenResearchAutoReboot='true'\n" not in before:
        raise ValueError('automatic reboot is not explicitly enabled; leave existing policy alone')
    scheduled=SCHEDULED
    if scheduled.exists():
        raise ValueError('scheduled shutdown needs separate attribution and explicit cancellation; preserve it')
    record=dict(started_utc=datetime.now(timezone.utc).isoformat(),host=host,
        configuration_path=str(CONFIG),configuration_previously_absent=True,
        previous_apt_configuration=before,previous_scheduled_shutdown=None,
        restore_required_at_goal_completion=True,updates_disabled=False,training_signalled=False,
        stage='installing_automatic_reboot_hold')
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as stream:json.dump(record,stream,indent=2)
    descriptor=os.open(CONFIG,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
    with os.fdopen(descriptor,'w') as stream:
        stream.write(CONTENT);stream.flush();os.fsync(stream.fileno())
    after=settings()
    record.update(effective_apt_configuration=after,configuration_content=CONFIG.read_text(),
        completed_utc=datetime.now(timezone.utc).isoformat(),stage='hold_installed_pending_verification')
    output.write_text(json.dumps(record,indent=2))
    if "ZenResearchAutoReboot='false'\n" not in after or scheduled.exists():
        raise ValueError('reboot hold precedence or absence of scheduled shutdown not verified')
    record['stage']='held';output.write_text(json.dumps(record,indent=2));return record


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    print(json.dumps(hold(parser.parse_args().output)),flush=True)
